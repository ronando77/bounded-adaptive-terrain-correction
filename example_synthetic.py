"""Run the unchanged algorithms on entirely generated 0.3 m terrain arrays."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'src'))
import numpy as np
import bounded_computation as method


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    lock = json.loads((HERE/'source_integrity.json').read_text(encoding='utf-8'))
    for record in lock:
        if digest(HERE/record['path']) != record['source_sha256']:
            raise AssertionError('Source differs from the audited manuscript implementation: '+record['path'])
    index = np.arange(-68, 69, dtype=float)
    x, y = np.meshgrid(0.3*index + 0.071, 0.3*index + 0.043)
    valid = np.ones(x.shape, dtype=bool)
    terrain = {
        'synthetic_gentle': 0.03*x - 0.02*y + 0.15*np.sin(x/4)*np.cos(y/5),
        'synthetic_relief': 0.35*x + 0.15*y + 1.2*np.sin(x/2.5)*np.cos(y/3.5),
        'synthetic_step': 0.05*x + np.where(x > 4.0, 2.0, -0.5),
    }
    output = []
    for name, height in terrain.items():
        args = (name, x, y, height, valid, 0.3)
        direct, _ = method.evaluate('full_prism_direct', args)
        for epsilon in (1.2e-5, 1.2e-4):
            fast, _ = method.fast_selective(args, epsilon, method.base.pixel_gravity_enclosures)
            sorted_result, _ = method.selective_prisms(args, epsilon)
            error = abs(fast['adaptive_mgal'] - direct['adaptive_mgal'])
            assert fast['budget_mgal'] == epsilon
            assert error <= epsilon
            assert error <= fast['model_bound_mgal'] + 1e-12
            assert fast['model_bound_mgal'] <= epsilon
            assert fast['exact_cells'] == sorted_result['exact_cells']
            assert abs(fast['adaptive_mgal'] - sorted_result['adaptive_mgal']) <= 1e-15
            output.append(dict(case=name, requested_budget_mgal=epsilon,
                direct_mgal=direct['adaptive_mgal'], adaptive_mgal=fast['adaptive_mgal'],
                absolute_difference_mgal=error, reported_bound_mgal=fast['model_bound_mgal'],
                exact_cells=fast['exact_cells'], covered_cells=fast['covered_cells'],
                sorted_cell_fraction=fast.get('sorted_cell_fraction'), budget_check=True,
                full_sort_equivalence_check=True))
    args = ('synthetic_forced_reserve_fallback', x, y, terrain['synthetic_relief'], valid, 0.3)
    fallback, _ = method.fast_selective(args, 1e-14, method.base.pixel_gravity_enclosures)
    full, _ = method.full_prism(args)
    assert fallback['budget_mgal'] == 0.0
    assert fallback['exact_cells'] == fallback['covered_cells']
    assert fallback['adaptive_mgal'] == full['adaptive_mgal']
    record = dict(status='PASS_SYNTHETIC_EXAMPLE', python=platform.python_version(), numpy=np.__version__,
                  data='Generated arrays only; no measured station or DTM is loaded.',
                  role='Executable code demonstration; not additional scientific validation.',
                  source_checks=len(lock), cases=output,
                  fallback=dict(requested_budget_mgal=1e-14, returned_budget_mgal=fallback['budget_mgal'],
                                exact_cells=fallback['exact_cells'], covered_cells=fallback['covered_cells'],
                                equality_to_full_wrapper=True),
                  floating_point_scope='Conservative padding and reserve; not a directed-rounding machine proof.')
    (HERE/'synthetic_example_results.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=record['status'], cases=len(output), fallback=True, source_checks=len(lock))))


if __name__ == '__main__':
    main()
