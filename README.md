# Bounded adaptive near-zone terrain correction

Numerical code accompanying *Adaptive Evaluation of LiDAR-Based Near-Zone Gravity Corrections under a Station-Level Error Budget*.

The method evaluates a prescribed digital terrain model (DTM) mass distribution. Each cell contributes either an inexpensive midpoint of its gravity-response interval or its analytical rectangular-prism value. The sum of the accepted interval half-widths controls approximation to the same discrete prism model. A preliminary threshold reduces the number of widths sorted while preserving the full-sort allocation in real arithmetic.

## Run the synthetic example

Python 3.12 is recommended. From the repository directory:

```sh
python -m pip install -r requirements.txt
python example_synthetic.py
```

The example requires NumPy only. It generates three artificial 0.3 m terrain windows (gentle relief, stronger relief and a step) and evaluates each with two budgets. It also exercises the protective full-station fallback by requesting a budget smaller than the numerical reserve. No raster, workbook or survey station file is read.

A successful run prints a JSON status containing `PASS_SYNTHETIC_EXAMPLE`, six normal cases and a successful fallback check. The generated details are written to `synthetic_example_results.json`, which is excluded from version control. The example checks:

- Observed differences from direct analytical summation against the requested budget and reported padded bound.
- Equal analytical cell counts and matching correction values for threshold-accelerated and full-sort allocation.
- Agreement of the forced fallback with the full-prism wrapper.
- The integrity of the included source files before calculation.

These synthetic checks demonstrate the numerical implementation. They do not reproduce the manuscript's field statistics or add field validation.

## Inputs and units

The calculation starts from DTM cell arrays. It does not process raw LiDAR points. An input tuple has the form:

```python
(station_id, x, y, height, valid, cell_size)
```

`x` and `y` are horizontal cell-centre coordinates relative to the station, in metres. `height` is terrain elevation minus the station reference-plane elevation, in metres. `valid` is a Boolean mask and `cell_size` is the square-cell width in metres. Arrays must have compatible shapes. All included cells must be valid and have finite heights. Horizontal coordinates and elevations must use consistent coordinate and height conventions before calculation.

The published high-level wrappers retain complete cell prisms whose centres lie within 20 m, with a uniform density of 2,200 kg/m³. These are explicit constants in `src/bounded_computation.py`. They are not runtime options of that wrapper. The lower-level prism and enclosure functions accept density as an argument. Changing radius, grid or density requires consistent inputs, recalculated intervals and a new assessment; the example does not validate such changes.

Corrections, budgets and error bounds are in mGal. The gravitational constant is in SI units, and `SI_TO_MGAL = 1e5` converts m/s² to mGal. The prism routines return terrain-correction magnitudes under the code's signed-height convention.

## Methods and error budget

Import the numerical wrapper after adding this repository's `src` directory to Python's import path, as shown in `example_synthetic.py`.

| Method | Role |
| --- | --- |
| `full_prism_direct` | Direct analytical summation on the matched complete-cell mass model; the same-grid comparison target. |
| `radial_fast_1p2e-5` | Threshold-accelerated allocation with a 0.000012 mGal budget. |
| `radial_fast_1p2e-4` | Threshold-accelerated allocation with a 0.00012 mGal budget. |
| `selective_1p2e-5` / `selective_1p2e-4` | Full-sort allocation controls at the same two budgets. |
| `full_prism` | Full-prism wrapper, also used by the protective fallback. |
| `adaptive_original` | Earlier subdomain model-selection implementation retained for source compatibility; this is not the allocation evaluated by the six main synthetic cases. |

The result includes `adaptive_mgal`, `model_bound_mgal`, `budget_mgal`, `covered_cells`, `exact_cells` and `prism_cell_fraction`. Accelerated allocation also reports `sorted_cell_fraction`. Here `exact` names an analytical prism evaluation; it does not mean exact floating-point arithmetic. If a requested budget cannot accommodate the numerical reserve, the full-prism wrapper returns its own zero approximation budget and bound. Keep the requested budget separately when recording a run.

The interval argument is established in real arithmetic. Double-precision padding and a station reserve are conservative numerical safeguards; they are not a directed-rounding enclosure of every operation in the complete prism kernel. The bound controls numerical approximation to the prescribed discrete terrain model. It does not bound errors in terrain elevation, density, station position or the physical gravity field.

## Numerical reference and data access

`full_prism_direct` evaluates the same grid as the bounded method. It is the appropriate target for checking allocation error. A calculation on a finer DTM is a separate **numerical reference**, not measured ground truth. Resolution, boundary treatment and represented masses must be documented before comparing different grids. The synthetic example uses generated 0.3 m arrays and does not contain the manuscript's 0.1 m reference DTM.

Survey stations, coordinates, observations, workbooks, measured DTMs, field figures and numerical field records are confidential and are not included. Reproduction of the reported field results requires the original authorized inputs and the study's field-processing instructions. Contact the corresponding author for access; release depends on the data owner's permission and applicable confidentiality conditions. Repository access does not itself grant permission to obtain or redistribute those materials.

## Source layout

- `src/bounded_computation.py`: matched full-prism, full-sort and threshold-accelerated allocation wrappers.
- `src/gravity_terrain_research/models/adaptive_bounded.py`: radial-kernel enclosures, cell prisms and the earlier subdomain implementation.
- `src/gravity_terrain_research/models/candidates.py`: unchanged numerical configuration and radial helper definitions needed by the retained import chain.
- `src/gravity_terrain_research/models/geometry.py`: numerical geometry helpers.
- `src/gravity_terrain_research/physics/formulae.py`: analytical terrain formulae and the prism kernel.
- `source_integrity.json`: hashes checked by the example.

The compact helper module omits survey-data readers, batch field-audit commands and checkpoint handling. The retained computational definitions and numerical rules are unchanged. No field-data loading dependency is required to run the example.

## License

No code license has been designated in this source package. An open-source license has not been inferred or added. Obtain the rights holder's permission for use or redistribution beyond permissions separately granted. NumPy remains subject to its own license; it is installed as a dependency and is not vendored here.
