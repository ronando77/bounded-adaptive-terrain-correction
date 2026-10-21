"""DTM-only local model selection with a certified gravity-approximation budget.

The target quantity bounded here is the mass of complete DTM cell prisms whose
centres belong to the 20 m neighbourhood, not unknown true terrain gravity.
No finest-grid or full-neighbourhood comparison result is an input.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .candidates import CandidateConfig, _fit_radial_or_constant, _radial_parts
from ..physics import formulae

ALGORITHM_VERSION='2.1.0-gravity-enclosure-refinement'


@dataclass(frozen=True)
class BoundedConfig:
    geometry: CandidateConfig=CandidateConfig()
    model_budget_mgal: float=0.0012
    require_gravity_bound: bool=True
    maximum_gravity_refinement_depth: int=2

    def validate(self):
        self.geometry.validate()
        if not math.isfinite(self.model_budget_mgal) or self.model_budget_mgal <= 0:
            raise ValueError('The model budget must be finite and positive.')
        if self.maximum_gravity_refinement_depth<0:raise ValueError('Refinement depth must be nonnegative.')


def vertical_kernel(r, height):
    """Stable 1/r - 1/sqrt(r*r+H*H), for r strictly positive."""
    r=np.asarray(r,dtype=float)
    h=np.abs(np.asarray(height,dtype=float))
    if np.any(r<=0):raise ValueError('Kernel bounds require positive horizontal distance.')
    length=np.hypot(r,h)
    return h*h/(r*length*(length+r))


def pixel_prisms(x,y,height,cell,density):
    half=cell/2
    h=np.asarray(height,float)
    return np.abs(formulae.prism_vertical_si(x-half,x+half,y-half,y+half,
        np.minimum(h,0),np.maximum(h,0),np.where(h>=0,density,-density)))*formulae.SI_TO_MGAL


def pixel_gravity_enclosures(x,y,height,cell,density):
    """Bounds by monotonicity in horizontal distance; singular cells exact."""
    x,y,h=np.asarray(x,float),np.asarray(y,float),np.asarray(height,float)
    if cell<=0 or density<=0:raise ValueError('Positive cell size and density required.')
    half=cell/2
    rlo=np.hypot(np.maximum(abs(x)-half,0),np.maximum(abs(y)-half,0))
    rhi=np.hypot(abs(x)+half,abs(y)+half)
    ordinary=rlo>0
    lo=np.zeros_like(h);hi=np.zeros_like(h)
    factor=formulae.G*density*formulae.SI_TO_MGAL*cell**2
    lo[ordinary]=factor*vertical_kernel(rhi[ordinary],h[ordinary])
    hi[ordinary]=factor*vertical_kernel(rlo[ordinary],h[ordinary])
    central=~ordinary
    if np.any(central):
        exact=pixel_prisms(x[central],y[central],h[central],cell,density)
        lo[central]=exact;hi[central]=exact
    return lo,hi,central


def compute_bounded_station(station_id,x,y,height,valid,cell,config=BoundedConfig()):
    config.validate();g=config.geometry
    x,y,h=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float),np.asarray(height,float))
    valid=np.asarray(valid,bool)&np.isfinite(h)
    radius=np.hypot(x,y)
    intended=radius<=g.radius_m
    if np.any(intended&~valid):raise ValueError('Missing DTM cells inside the intended circle.')
    if not np.any(intended):raise ValueError('Empty near-zone domain.')
    x,y,h,r=x[intended],y[intended],h[intended],radius[intended]
    theta=np.mod(np.arctan2(y,x),2*math.pi)
    n=len(h);angle=2*math.pi/g.sector_count
    low,high,central=pixel_gravity_enclosures(x,y,h,cell,g.density_kg_m3)
    membership=np.full(n,-1,dtype=int)
    rows=[]
    for sector in range(g.sector_count):
        sector_mask=(theta>=sector*angle)&(theta<(sector+1)*angle)
        if not np.any(sector_mask):raise ValueError('An intended azimuth sector contains no cells.')
        pending=[(a,b,0) for a,b in _radial_parts(r[sector_mask],h[sector_mask],g)]
        while pending:
            start,end,depth=pending.pop(0)
            mask=sector_mask&(r>=start)&(r<end)
            count=int(mask.sum())
            if not count:continue
            if np.any(membership[mask]>=0):raise AssertionError('Overlapping subdomains.')
            domain_id=len(rows);membership[mask]=domain_id
            hr,rr=h[mask],r[mask]
            relief=float(np.quantile(hr,.95)-np.quantile(hr,.05))
            limit=max(g.geometry.fit_rmse_floor_m,g.geometry.fit_rmse_relief_fraction*max(relief,.1))
            intercept_limit=max(g.intercept_limit_floor_m,g.intercept_limit_relief_fraction*max(relief,.1))
            fit_rmse,constant_rmse,beta=_fit_radial_or_constant(rr,hr)
            actual_cone_rmse=float(np.sqrt(np.mean((hr-beta[0]*rr)**2)))
            lower,upper=float(np.sum(low[mask])),float(np.sum(high[mask]))
            budget=config.model_budget_mgal*count/n
            # Directed padding is numerical protection, not a tunable physical allowance.
            pad=64*np.finfo(float).eps*max(1.0,abs(lower),abs(upper))
            lower=max(0.0,lower-pad);upper+=pad
            cone_ok=(actual_cone_rmse<=limit and abs(beta[1])<=intercept_limit
                and abs(beta[0])>=g.minimum_cone_slope
                and actual_cone_rmse<=g.cone_preference_ratio*max(constant_rmse,1e-8))
            constant_ok=constant_rmse<=limit
            proposals=[]
            if cone_ok:
                slope=float(beta[0])
                value=formulae.conical_sector_correction_mgal(min(end,g.radius_m),math.atan(abs(slope)),g.density_kg_m3,angle)
                value-=formulae.conical_sector_correction_mgal(start,math.atan(abs(slope)),g.density_kg_m3,angle)
                proposals.append(('conical_sector',value,actual_cone_rmse))
            if constant_ok:
                value=formulae.sector_correction_mgal(start,min(end,g.radius_m),float(np.median(hr)),g.density_kg_m3,angle)
                proposals.append(('constant_sector',value,constant_rmse))
            chosen='grid_prism';value=math.nan;bound=0.0;rmse=min(actual_cone_rmse,constant_rmse)
            reason='irregular_geometry';rejected=0
            for model,candidate,error in proposals:
                candidate_bound=max(abs(candidate-lower),abs(candidate-upper))
                if not config.require_gravity_bound or candidate_bound<=budget:
                    chosen,value,bound,rmse=model,candidate,candidate_bound,error
                    reason='geometry_and_gravity_bound' if config.require_gravity_bound else 'geometry_only_ablation'
                    break
                rejected+=1;reason='gravity_bound_requires_prisms'
            if chosen=='grid_prism' and proposals and config.require_gravity_bound and depth<config.maximum_gravity_refinement_depth:
                # Equal nominal area partitions separate near/far kernel sensitivity;
                # the gravity budget is still assigned by actual cell count.
                middle=math.sqrt((start**2+min(end,g.radius_m)**2)/2)
                inner_count=int(np.sum(rr<middle))
                if min(inner_count,count-inner_count)>=15 and min(middle-start,min(end,g.radius_m)-middle)>=cell:
                    membership[mask]=-1
                    pending[:0]=[(start,middle,depth+1),(middle,end,depth+1)]
                    continue
            rows.append(dict(station_id=str(station_id),subdomain_id=domain_id,azimuth_sector=sector,
                radius_min_m=start,radius_max_m=min(end,g.radius_m),gravity_refinement_depth=depth,cell_count=count,selected_model=chosen,
                selection_reason=reason,normal_prism_selection=chosen=='grid_prism' and not proposals,
                safety_prism_selection=chosen=='grid_prism' and bool(proposals),rejected_analytic_proposals=rejected,
                radial_slope=float(beta[0]),intercept_m=float(beta[1]),median_height_m=float(np.median(hr)),
                free_intercept_rmse_m=fit_rmse,actual_cone_rmse_m=actual_cone_rmse,constant_rmse_m=constant_rmse,
                selected_geometry_rmse_m=rmse,geometry_limit_m=limit,intercept_limit_m=intercept_limit,
                gravity_lower_mgal=lower,gravity_upper_mgal=upper,budget_mgal=budget,model_bound_mgal=bound,
                contribution_mgal=value,central_exact_cells=int(central[mask].sum())))
    if np.any(membership<0):raise AssertionError('Incomplete subdomain coverage.')
    # Only selected numerical domains undergo full cell-prism integration.
    numerical=np.array([row['selected_model']=='grid_prism' for row in rows])[membership]
    if np.any(numerical):
        values=pixel_prisms(x[numerical],y[numerical],h[numerical],cell,g.density_kg_m3)
        summed=np.bincount(membership[numerical],weights=values,minlength=len(rows))
        for row in rows:
            if row['selected_model']=='grid_prism':row['contribution_mgal']=float(summed[row['subdomain_id']])
    result=dict(station_id=str(station_id),cell_size_m=float(cell),adaptive_mgal=math.fsum(row['contribution_mgal'] for row in rows),
        model_bound_mgal=math.fsum(row['model_bound_mgal'] for row in rows),budget_mgal=config.model_budget_mgal,
        subdomain_count=len(rows),cone_count=sum(row['selected_model']=='conical_sector' for row in rows),
        constant_count=sum(row['selected_model']=='constant_sector' for row in rows),prism_count=sum(row['selected_model']=='grid_prism' for row in rows),
        normal_prism_count=sum(row['normal_prism_selection'] for row in rows),safety_prism_count=sum(row['safety_prism_selection'] for row in rows),
        prism_cell_fraction=float(numerical.mean()),covered_cells=n,status='ok',algorithm_version=ALGORITHM_VERSION)
    if config.require_gravity_bound and result['model_bound_mgal']>config.model_budget_mgal+1e-12:
        raise AssertionError('Station error budget exceeded.')
    return result,rows
