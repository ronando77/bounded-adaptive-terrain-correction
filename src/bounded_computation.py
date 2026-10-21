"""Published bounded computation, extracted without changing numerical rules."""
from __future__ import annotations
import math
import numpy as np
from gravity_terrain_research.models import adaptive_bounded as base
from gravity_terrain_research.physics import formulae
EPS = np.finfo(float).eps
CFG = base.BoundedConfig()

def clipped(args):
    sid, x, y, h, valid, cell = args
    x, y, h = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(h, float))
    keep = np.hypot(x, y) <= 20.
    if not keep.any() or not np.all(np.asarray(valid, bool)[keep] & np.isfinite(h[keep])):
        raise ValueError('Invalid near-zone coverage')
    return x[keep], y[keep], h[keep], cell

def full_prism(args):
    sid, x, y, h, valid, cell = args
    value = float(formulae.numerical_grid_correction_mgal(x, y, h, valid & (np.hypot(x, y) <= 20.), cell, 2200.))
    n = int(np.count_nonzero(np.hypot(x, y) <= 20.))
    return dict(adaptive_mgal=value, model_bound_mgal=0., budget_mgal=0., covered_cells=n,
                exact_cells=n, prism_cell_fraction=1.), []

def selective_prisms(args, epsilon):
    """Midpoints for narrow cells; exact prisms remove the widest uncertainties.

    For each cell, L_i <= g_i <= U_i. With m_i=(L_i+U_i)/2 and
    b_i=(U_i-L_i)/2, |sum(g)-[sum_exact(g)+sum_approx(m)]| <= sum_approx(b).
    Sorting b ascending maximizes the number approximated under a common cost
    per exact cell and this additive interval bound. It is not a globally
    optimal runtime claim and does not change the input mass representation.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Positive finite budget required')
    x, y, h, cell = clipped(args)
    low, high, central = base.pixel_gravity_enclosures(x, y, h, cell, 2200.)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
        raise ArithmeticError('Invalid response interval')
    n = len(h)
    middle = low + .5 * (high - low)
    # Protect computed ordinary-cell endpoints; existing singular prism kernel
    # remains the same floating-point target used by the production method.
    pad = 64 * np.finfo(float).eps * np.maximum(1., np.maximum(abs(low), abs(high)))
    width = .5 * (high - low) + pad
    width[central] = 0.
    rounding = 1e-12 + 16 * n * np.finfo(float).eps * max(1., float(np.sum(abs(middle))))
    available = max(0., epsilon - rounding)
    order = np.argsort(width, kind='stable')
    cumulative = np.cumsum(width[order])
    count = int(np.searchsorted(cumulative, available, side='right'))
    approx = order[:count]
    exact = order[count:]
    # Direct sum verifies the chosen remainder rather than subtracting two
    # potentially huge near-singular interval sums.
    while count and float(np.sum(width[approx])) > available:
        count -= 1
        approx, exact = order[:count], order[count:]
    correction = float(np.sum(middle[approx]))
    if len(exact):
        correction += float(np.sum(base.pixel_prisms(x[exact], y[exact], h[exact], cell, 2200.)))
    computed_bound = float(np.sum(width[approx])) + rounding
    if computed_bound > epsilon:
        # Very small budgets may be below conservative floating-point padding.
        return full_prism(args)
    exact_count = int(len(exact) + np.count_nonzero(central[approx]))
    return dict(adaptive_mgal=correction, model_bound_mgal=computed_bound,
                budget_mgal=epsilon, covered_cells=n, exact_cells=exact_count,
                prism_cell_fraction=exact_count / n), []

def fast_selective(args, epsilon, enclosure):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Positive finite budget required')
    x, y, h, cell = clipped(args)
    low, high, central = enclosure(x, y, h, cell, 2200.)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
        raise ArithmeticError('Invalid response interval')
    n = len(h)
    middle = low+.5*(high-low)
    pad = 64*EPS*np.maximum(1., np.maximum(abs(low), abs(high)))
    width = .5*(high-low)+pad
    width[central] = 0.
    rounding = 1e-12+16*n*EPS*max(1., float(np.sum(abs(middle))))
    available = max(0., epsilon-rounding)
    approx = width <= available/n
    used = float(np.sum(width[approx]))
    remaining = np.flatnonzero(~approx)
    # All initially accepted elements precede these in the full sorted sequence.
    order = remaining[np.argsort(width[remaining], kind='stable')]
    cumulative = np.cumsum(width[order])
    count = int(np.searchsorted(cumulative, max(0., available-used), side='right'))
    approx[order[:count]] = True
    while count and float(np.sum(width[approx])) > available:
        count -= 1
        approx[order[count]] = False
    bound = float(np.sum(width[approx]))+rounding
    if bound > epsilon:
        return full_prism(args)
    correction = float(np.sum(middle[approx]))
    exact = ~approx
    if np.any(exact):
        correction += float(np.sum(base.pixel_prisms(x[exact], y[exact], h[exact], cell, 2200.)))
    exact_count = int(exact.sum()+central[approx].sum())
    return dict(adaptive_mgal=correction, model_bound_mgal=bound, budget_mgal=epsilon,
                covered_cells=n, exact_cells=exact_count, prism_cell_fraction=exact_count/n,
                sorted_cell_fraction=len(remaining)/n), []

METHODS = ['full_prism', 'full_prism_direct', 'adaptive_original', 'selective_1p2e-4', 'selective_1p2e-5', 'radial_fast_1p2e-4', 'radial_fast_1p2e-5']
BUDGETS = dict(zip(METHODS, [0., 0., .0012, .00012, .000012, .00012, .000012]))

def evaluate(method, args):
    if method == 'full_prism_direct':
        x,y,h,cell = clipped(args)
        value = float(np.sum(base.pixel_prisms(x,y,h,cell,2200.)))
        n = len(h)
        return dict(adaptive_mgal=value,model_bound_mgal=0.,budget_mgal=0.,covered_cells=n,exact_cells=n,prism_cell_fraction=1.), []
    if method == 'full_prism':
        return full_prism(args)
    if method == 'adaptive_original':
        return base.compute_bounded_station(*args, CFG)
    if method.startswith('selective_') and method in BUDGETS:
        return selective_prisms(args, BUDGETS[method])
    if method.startswith('radial_fast_') and method in BUDGETS:
        return fast_selective(args, BUDGETS[method], base.pixel_gravity_enclosures)
    raise ValueError(method)
