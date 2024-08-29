# Continent contouring

Included are some workflows that use continent contouring from [`GPlately`](https://github.com/GPlates/gplately). This essentially amounts to creating a `ContinentContouring` object using rotation files, some continent/craton features (polygons), a contour resolution to determine how finely tessellated the contour outlines should be, a buffer/gap threshold to expand continents outward, and two area thresholds to separately exclude small continental islands and small oceanic islands. It will then reconstruct the polygons to an `age` and contour them into continents. This will output the continent contours (as *polyline* continental-oceanic boundaries) and/or continent masks (a 2D NumPy boolean array of continental crust at each age):

```
from gplately.ptt.continent_contours import ContinentContouring

continent_contouring = ContinentContouring(...)
...
contoured_continents = continent_contouring.get_contoured_continents(age)
continent_mask = continent_contouring.get_continent_mask(age)
```

The workflows then use these contours and masks in different ways.

## Installation

### Dependencies

The following Python packages are required:

- [`gplately`](https://github.com/GPlates/gplately)
- [`numpy`](http://numpy.org)
- [`pygplates`](http://gplates.org/docs/pygplates/pygplates_getting_started.html#installation)

You can install these with conda:

```
conda create -n <conda-environment> -c conda-forge gplately numpy pygplates
conda activate <conda-environment>
```

...where `<conda-environment>` should be replaced with the name of your conda environment.

## Documentation

Current workflows include:

### create_passive_margins.py

Creates passive margins through time by excluding those parts of contours that are considered to be active margins (that are within a threshold distance from subduction zones).

To run the workflow you'll first need to copy the [Merdith et al (2021) 1Ga plate model files](https://www.earthbyte.org/webdav/ftp/Data_Collections/Merdith_etal_2021_ESR/SM2-Merdith_et_al_1_Ga_reconstruction_v1.1.zip) into `models/Merdith_etal_2021_Published/`.

> _Note:_ You can use a different plate model by changing the parameters `rotation_features`, `topology_features` and `continent_features` to refer to files that you provide. And you can change the time range.

You can then run the workflow with `python create_passive_margins.py`. You should end up with a files `output/passive_margin_features.gpmlz` containing the passive margin features, `output/subduction_zone_features.gpmlz` containing the subduction zones, `output/continent_contour_features.gpmlz` containing the continental-oceanic boundaries (COBs) as *polylines* (these are all the contours around continental crust), and a NetCDF grid at each time `output/continent_mask_<time>.nc` representing the mask of continental crust at that time.

Parameters at the top of `create_passive_margins.py` control its behaviour. The time range is specified with `start_time`, `end_time` and `time_interval`. The maximum distance of an active margin from a subduction zone (in kms) is specified with `max_distance_of_subduction_zone_from_active_margin_kms`. The grid spacing (in degrees) between points in the grid used for contouring/aggregrating blocks of continental polygons is specified with `continent_contouring_point_spacing_degrees`. The parameter `continent_contouring_area_threshold_steradians` (which can also be a function of time `continent_contouring_area_threshold_steradians(time)`) determines if contoured continents smaller than this will be excluded. The parameter `continent_exclusion_area_threshold_steradians` (which can also be a function of time) determines if oceanic holes (in continents) smaller than this will be excluded. The parameter `continent_contouring_buffer_and_gap_distance_radians` (which can also be either a function of time or a function of both time and a specific contoured continent) will expand continents ocean-ward by a buffer of this distance. The parameter `continent_separation_distance_threshold_radians` (which can also be a function of time) determines the distance above which continents become separated. And finally, the `use_all_cpus` parameter controls CPU usage (whether to use multiple CPUs and how many).

All the passive margins found for all times end up in a single file (similarly for subduction zones and continent contours). Each passive margin feature only exists at its reconstruction time (+/- 0.5 * `time_interval`). And you don’t need a rotation file to view these through time (they don’t have plate IDs) since they are meant to be reconstructed snapshots (from 0Ma to 1000Ma).

> _Note:_ When running the script you’ll get lots of temporary files, but they’ll get cleaned up when the script finishes (provided it’s not interrupted).

### create_passive_margins_from_non_contoured.py

This workflow does not actually use continent *contouring*. Instead it's a version of `create_passive_margins.py` that reconstructs continent boundaries (instead of contouring reconstructed continental polygons). Because of this, the continent features should be continent **boundaries** (otherwise anything *inside* a continent will mistakenly get picked up as a passive margin).

However, the only output files are `output/passive_margin_features.gpmlz` containing the passive margin features and `output/subduction_zone_features.gpmlz` containing the subduction zones. But, like `create_passive_margins.py`, all the passive margins found for all times end up in a single file (similarly for subduction zones), and these are snapshots (ie, don't need a rotation model to view).
