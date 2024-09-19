from functools import partial
import gplately
import math
import multiprocessing
import numpy as np
import os
import os.path
import pygplates
from gplately.ptt import continent_contours


#############################
# Start of input parameters #
#############################

# Location of Cao2024 1.8Ga plate model.
model_dir = os.path.join('/Volumes/Carbon_backup/CONTOURS/Sep11-Cao2024-Original', 'models/1.8Ga_model_2024_07_25')

# Rotation files (relative to input directory).
rotation_features = [pygplates.FeatureCollection(os.path.join(model_dir, file)) for file in (
    'optimisation/1800_1000_rotfile_20240725_run3.rot',
    'optimisation/1000_0_rotfile_20240725_run3.rot',
    #'optimisation/no_net_rotation_model_20240725_run3.rot',
)]

# The reference frame to generate the output files relative to.
anchor_plate_id = 0

# Topology features (absolute file paths).
#
# Only include those GPML files that are used for topologies.
topology_features = [pygplates.FeatureCollection(os.path.join(model_dir, file)) for file in (
    '1800-1000_plate_boundaries.gpml',
    '250-0_plate_boundaries.gpml',
    '410-250_plate_boundaries.gpml',
    '1000-410-Convergence.gpml',
    '1000-410-Divergence.gpml',
    '1000-410-Topologies.gpml',
    '1000-410-Transforms.gpml',
    'TopologyBuildingBlocks.gpml',
)]

# Continent polygon features (absolute file paths).
continent_features = [pygplates.FeatureCollection(os.path.join(model_dir, file)) for file in (
    'shapes_continents_Cao.gpmlz',
)]

# Time range.
start_time = 0
end_time = 1800
time_interval = 1
times = np.arange(start_time, end_time + 0.5 * time_interval, time_interval)

# Use all CPUs.
#
# If False then use a single CPU.
# If True then use all CPUs (cores) - and make sure you don't interrupt the process.
# If a positive integer then use that specific number of CPUs (cores).
#
#use_all_cpus = False
#use_all_cpus = 4
use_all_cpus = True

# Maximum distance of an active margin from a subduction zone.
#
# If a contoured continent segment is near any subduction zone (ie, within this distance) then it's an active margin (ie, not a passive margin).
max_distance_of_subduction_zone_from_active_margin_kms = 500
max_distance_of_subduction_zone_from_active_margin_radians = max_distance_of_subduction_zone_from_active_margin_kms / pygplates.Earth.mean_radius_in_kms

# The directory containing the output files.
output_dir = 'output'

# The grid spacing (in degrees) between points in the grid used for contouring/aggregrating blocks of continental polygons.
continent_contouring_point_spacing_degrees = 0.1

# Optional parameter specifying a minimum area threshold (in square radians) for including contoured continents.
#
# Contoured continents with area smaller than this threshold will be excluded.
# If this parameter is not specified then no area threshold is applied.
#
# Can also be a function (accepting time in Ma) and returning the area threshold.
#
# Note: Units here are for normalised sphere (ie, steradians or square radians) so full Earth area is 4*pi.
#       So 0.1 covers an area of approximately 4,000,000 km^2 (ie, 0.1 * 6371^2, where Earth radius is 6371km).
#       Conversely 4,000,000 km^2 is equivalent to (4,000,000 / 6371^2) steradians.
continent_contouring_area_threshold_square_kms = 0
continent_contouring_area_threshold_steradians = continent_contouring_area_threshold_square_kms / (pygplates.Earth.mean_radius_in_kms * pygplates.Earth.mean_radius_in_kms)

# Optional parameter specifying a minimum area threshold (in square radians) for contours that exclude continental crust.
#
# Polygon contours that exclude continental crust and have an area smaller than this threshold will be excluded
# (meaning they will now *include* continental crust, thus removing the contour).
# This is useful for removing small holes inside continents.
# If this parameter is not specified then no area threshold is applied.
#
# Can also be a function (accepting time in Ma) and returning the area threshold.
#
# Note: Units here are for normalised sphere (ie, steradians or square radians) so full Earth area is 4*pi.
#       So 0.1 covers an area of approximately 4,000,000 km^2 (ie, 0.1 * 6371^2, where Earth radius is 6371km).
#       Conversely 4,000,000 km^2 is equivalent to (4,000,000 / 6371^2) steradians.
def continent_exclusion_area_threshold_steradians(time):
    if time > 1450:
        continent_exclusion_area_threshold_square_kms = 3500000

    elif time > 1400:
        # Linearly interpolate between 1450 and 1400 Ma.
        interp = float(time - 1400) / (1450 - 1400)
        continent_exclusion_area_threshold_square_kms = interp * 3500000 + (1 - interp) * 900000
    
    elif time > 200:
        continent_exclusion_area_threshold_square_kms = 900000
    else:
        continent_exclusion_area_threshold_square_kms = 0

    continent_exclusion_area_threshold_steradians = continent_exclusion_area_threshold_square_kms / (pygplates.Earth.mean_radius_in_kms * pygplates.Earth.mean_radius_in_kms)
    return continent_exclusion_area_threshold_steradians
    
# Optional parameter specifying the distance threshold (in radians) above which continents are separated.
#
# Any continent polygons separated by a distance that is less than this threshold will become part of the same continent.
#
# Can also be a function (accepting time in Ma) and returning the distance threshold.
#
# Note: Units here are for normalised sphere (ie, radians).
#       So 1.0 radian is approximately 6371 km (where Earth radius is 6371 km).
#       Also 1.0 degree is approximately 110 km.
#
#continent_separation_distance_threshold_kms = 0
#continent_separation_distance_threshold_radians = continent_separation_distance_threshold_kms / pygplates.Earth.mean_radius_in_kms
def continent_separation_distance_threshold_radians(time):
    if time > 1450:
        # At times >1450, we need higher sep distance because this will make sure that 
        continent_separation_distance_threshold_radians = 0.30 #1 #continent_contours.DEFAULT_CONTINENT_SEPARATION_DISTANCE_THRESHOLD_RADIANS
    elif time > 1400:
        # Linearly interpolate between 1450 and 1400 Ma.
        interp = float(time - 1400) / (1450 - 1400)
        continent_separation_distance_threshold_radians = interp * 0.30 + (1 - interp) * 0.001
    elif time > 200:
        continent_separation_distance_threshold_radians = 0.001
    else:
        continent_separation_distance_threshold_radians = 0
    return continent_separation_distance_threshold_radians


# Optional parameter specifying a distance (in radians) to expand contours ocean-ward - this also
# ensures small gaps between continents are ignored during contouring.
#
# The continent(s) will be expanded by a buffer of this distance (in radians) when contouring/aggregrating blocks of continental polygons.
# If this parameter is not specified then buffer expansion is not applied.
#
# This parameter can also be a function (that returns the distance).
# The function can have a single function argument, accepting time (in Ma).
# Or it can have two function arguments, with the second accepting the contoured continent (a 'gplately.ptt.continent_contours.ContouredContinent' object)
# of the (unexpanded) contoured continent that the buffer/gap distance will apply to.
# Or it can have three function arguments, with the third accepting a list of reconstructed polygons ('pygplates.ReconstructedFeatureGeometry' objects)
# used to contour the (unexpanded) contoured continent that the buffer/gap distance will apply to.
# Hence a function with *two* arguments means a different buffer/gap distance can be specified for each contoured continent (eg, based on its area).
# And a function with *three* arguments can also use the feature properties (eg, plate ID) of the reconstructed polygons in the contoured continent.
#
# Note: Units here are for normalised sphere (ie, radians).
#       So 1.0 radian is approximately 6371 km (where Earth radius is 6371 km).
#       Also 1.0 degree is approximately 110 km.
class ContinentContouringBufferAndGapDistanceRadians(object):


    #From 1800 to 1400 Ma  use 3.25 degrees.
    #From 1400 to 1000 Ma linearly interpolate from 3.25 to 2.25 degrees.
    #From 1000 to 410 Ma  keep 2.25 degrees constant.
    #From 410 to 200 linearly interpolate between 2.25 degrees to 0 degrees, with the exception of the plate IDs in the attached list, which get interpolated from 2.25deg at 410 Ma to 1.0 deg at 200 Ma.
    #From 200-0 we use 0 degrees, except for the plate ID list for which we use 1.0 deg.

    # One distance for time interval [1000, 300] and another for time interval [200, 0].
    # And linearly interpolate between them over the time interval [300, 200].
    pre_pangea_distance_radians = math.radians(2.25)  # convert degrees to radians
    post_pangea_distance_radians = math.radians(0.0)  # convert degrees to radians
    
    # Plate IDs matching the post-pangea COBs.
    post_pangea_COB_plate_IDs = [520, 505, 506, 50601, 50602, 590, 591, 61601, 616, 606, 603, 7250, 60302, 64701, 647, 673, 67350, 614, 657, 456, 457, 467, 601, 60201, 612, 61404]

    def __call__(self, time, contoured_continent, continent_feature_polygons):
        if time >= 1400:
            buffer_and_gap_distance_radians = math.radians(3.25)
        elif time >= 1000:
            interp = float(time - 1000) / (1400 - 1000)
            buffer_and_gap_distance_radians = interp * math.radians(3.25) + (1 - interp) * self.pre_pangea_distance_radians
        elif time >= 410:
            buffer_and_gap_distance_radians = self.pre_pangea_distance_radians

        # From 410 to 200 linearly interpolate between 2.25 degrees to 0 degrees, with the exception of the plate IDs in the attached list, which get interpolated 
        # from 2.25deg at 410 Ma to 1.0 deg at 200 Ma.
        elif time >= 200:
            interp = float(time - 200) / (410 - 200)
            if self._contoured_continent_has_post_pangea_COBs(continent_feature_polygons):
                buffer_and_gap_distance_radians = interp * math.radians(2.25) + (1 - interp) * math.radians(1.0)
            else:
                buffer_and_gap_distance_radians = interp * math.radians(2.25) + (1 - interp) * self.post_pangea_distance_radians
        # From 200-0 we use 0 degrees, except for the plate ID list for which we use 1.0 deg.
        else:  # time <= 200 ...
            if self._contoured_continent_has_post_pangea_COBs(continent_feature_polygons):
                buffer_and_gap_distance_radians = math.radians(1.0)
            else:
                buffer_and_gap_distance_radians = self.post_pangea_distance_radians
        
        # Area of the contoured continent.
        area_steradians = contoured_continent.get_area()

        # Linearly reduce the buffer/gap distance for contoured continents with area smaller than 1 million km^2.
        area_threshold_square_kms = 500000
        area_threshold_steradians = area_threshold_square_kms / (pygplates.Earth.mean_radius_in_kms * pygplates.Earth.mean_radius_in_kms)
        if area_steradians < area_threshold_steradians:
            buffer_and_gap_distance_radians *= area_steradians / area_threshold_steradians

        return buffer_and_gap_distance_radians

    def _contoured_continent_has_post_pangea_COBs(self, continent_feature_polygons):
        # Return true if any polygon in the current contoured continent has a plate ID associated with the post-pangea COBs.
        for continent_feature_polygon in continent_feature_polygons:
            if (continent_feature_polygon.get_feature().get_reconstruction_plate_id() in self.post_pangea_COB_plate_IDs):
                return True
        
        return False


#continent_contouring_buffer_and_gap_distance_radians = ContinentContouringBufferAndGapDistanceRadians()


# Optional parameter specifying a distance (in radians) to expand each individual continental polygon ocean-ward - this also
# ensures small gaps between continents are ignored during contouring.
#
# NOTE: This is similar to 'continent_contouring_buffer_and_gap_distance_radians' except it applies to each continental polygon
#       (instead of applying to each aggregate block of continental polygons forming a continent contour).
#
# The continent polygons will be expanded by a buffer of this distance (in radians).
# If this parameter is not specified then buffer expansion is not applied (to continental polygons).
#
# This parameter can also be a function (that returns the distance).
# The function can have a single function argument, accepting time (in Ma).
# Or it can have two function arguments, with the second accepting the reconstructed continental feature polygon
# (a 'pygplates.ReconstructedFeatureGeometry' object) that the buffer/gap distance will apply to.
# Hence a function with *two* arguments means a different buffer/gap distance can be specified for each continental polygon.
# For example, you can use its feature properties (eg, plate ID), and/or its reconstructed polygon (eg, area).
#
# Note: Units here are for normalised sphere (ie, radians).
#       So 1.0 radian is approximately 6371 km (where Earth radius is 6371 km).
#       Also 1.0 degree is approximately 110 km.
#
# NOTE: This cannot be specified if 'continent_contouring_buffer_and_gap_distance_radians' is specified.
#       You can only specify one or the other (or neither).
class ContinentPolygonBufferAndGapDistanceRadians(object):
    # One distance for time interval [1000, 300] and another for time interval [200, 0].
    # And linearly interpolate between them over the time interval [300, 200].
    pre_pangea_distance_radians = math.radians(2.25)  # convert degrees to radians
    post_pangea_distance_radians = math.radians(0.0)  # convert degrees to radians
    # Plate IDs matching the post-pangea COBs.
    post_pangea_COB_plate_IDs = [520, 505, 506, 50601, 50602, 590, 591, 61601, 616, 606, 603, 7250, 60302, 64701, 647, 673, 67350, 614, 657, 456, 457, 467, 601, 60201, 612, 61404]

    ignore_ids = [8016, 8013] #Lachlan West, Rocky Cape

    def __call__(self, time, continent_feature_polygon):
        if continent_feature_polygon.get_feature().get_reconstruction_plate_id() in self.ignore_ids:
                buffer_and_gap_distance_radians = self.post_pangea_distance_radians
        else:
            if time >= 1400:
                buffer_and_gap_distance_radians = math.radians(3.25)
            elif time >= 1000:
                interp = float(time - 1000) / (1400 - 1000)
                buffer_and_gap_distance_radians = interp * math.radians(3.25) + (1 - interp) * self.pre_pangea_distance_radians
            elif time >= 410:
                buffer_and_gap_distance_radians = self.pre_pangea_distance_radians

            # From 410 to 200 linearly interpolate between 2.25 degrees to 0 degrees, with the exception of the plate IDs in the attached list, which get interpolated 
            # from 2.25deg at 410 Ma to 1.0 deg at 200 Ma.
            elif time >= 200:
                interp = float(time - 200) / (410 - 200)
                if continent_feature_polygon.get_feature().get_reconstruction_plate_id() in self.post_pangea_COB_plate_IDs:
                    buffer_and_gap_distance_radians = interp * math.radians(2.25) + (1 - interp) * math.radians(1.0)
                else:
                    buffer_and_gap_distance_radians = interp * math.radians(2.25) + (1 - interp) * self.post_pangea_distance_radians
            # From 200-0 we use 0 degrees, except for the plate ID list for which we use 1.0 deg.        
            else:  # time <= 200 ...
                if continent_feature_polygon.get_feature().get_reconstruction_plate_id() in self.post_pangea_COB_plate_IDs:
                    buffer_and_gap_distance_radians = math.radians(1.0)
                else:
                    buffer_and_gap_distance_radians = self.post_pangea_distance_radians

        return buffer_and_gap_distance_radians

continent_polygon_buffer_and_gap_distance_radians = ContinentPolygonBufferAndGapDistanceRadians()


###########################
# End of input parameters #
###########################


# Create output directory if it doesn't exist.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Rotation model.
rotation_model = pygplates.RotationModel(rotation_features, default_anchor_plate_id=anchor_plate_id)

# Create a ContinentContouring object.
continent_contouring = continent_contours.ContinentContouring(
        rotation_model,
        continent_features,
        continent_contouring_point_spacing_degrees,
        continent_contouring_area_threshold_steradians=continent_contouring_area_threshold_steradians,
        # Cannot be specified if 'continent_polygon_buffer_and_gap_distance_radians' is specified...
        continent_contouring_buffer_and_gap_distance_radians=None,
        continent_exclusion_area_threshold_steradians=continent_exclusion_area_threshold_steradians,
        continent_separation_distance_threshold_radians=continent_separation_distance_threshold_radians,
        # Cannot be specified if 'continent_contouring_buffer_and_gap_distance_radians' is specified...
        continent_polygon_buffer_and_gap_distance_radians=continent_polygon_buffer_and_gap_distance_radians)

# Find passive margins at the specified time.
def find_passive_margins(
        time):
    print('time:', time)
    
    continent_contour_features = []
    passive_margin_features = []
    subduction_zone_features = []
    
    # Resolve the topological plate polygons for the current time.
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, time, shared_boundary_sections)
    
    # Get the resolved subduction zone segments.
    subduction_zone_lines = []
    for shared_boundary_section in shared_boundary_sections:
        if shared_boundary_section.get_feature().get_feature_type() == pygplates.FeatureType.gpml_subduction_zone:
            shared_sub_segment_lines = [shared_sub_segment.get_resolved_geometry() for shared_sub_segment in shared_boundary_section.get_shared_sub_segments()]
            subduction_zone_lines.extend(shared_sub_segment_lines)
            # Also save subduction zone lines as features (so we can later save them to a file for debugging).
            subduction_zone_feature = pygplates.Feature()
            subduction_zone_feature.set_geometry(shared_sub_segment_lines)
            subduction_zone_feature.set_valid_time(time + 0.5 * time_interval - 1e-4,  # epsilon to avoid overlap at interval boundaries
                                                   time - 0.5 * time_interval)
            subduction_zone_features.append(subduction_zone_feature)

    # Get the continent mask and the continent contours at the current time.
    continent_mask, contoured_continents = continent_contouring.get_continent_mask_and_contoured_continents(time)

    # Write out the continent mask as NetCDF.
    #
    continent_mask_filename = os.path.join(output_dir, 'continent_mask_{}.nc'.format(time))
    # Note that we need to convert the boolean mask grid to a non-boolean number type for NetCDF (and it seems floating-point for gplately).
    continent_mask_grid = continent_mask.astype('float')
    gplately.grids.write_netcdf_grid(continent_mask_filename, continent_mask_grid)
    
    # Convert all continent contour geometries into features.
    for contoured_continent in contoured_continents:
        for continent_contour_geometry in contoured_continent.get_contours():
            continent_contour_feature = pygplates.Feature()
            continent_contour_feature.set_geometry(continent_contour_geometry)
            continent_contour_feature.set_valid_time(time + 0.5 * time_interval - 1e-4,  # epsilon to avoid overlap at interval boundaries
                                                     time - 0.5 * time_interval)
            continent_contour_features.append(continent_contour_feature)
    
    # Find passive margins along continent contours by removing active margins (contoured segments close to a subduction zone).
    passive_margin_geometries = []
    for contoured_continent in contoured_continents:
        for contour_polyline in contoured_continent.get_contours():
            # Add any passive margins found on the current contour.
            _find_passive_margin_geometries_on_contour(passive_margin_geometries,
                                                       subduction_zone_lines,
                                                       contour_polyline.get_segments())
    
    # Convert any passive margin geometries found into features.
    for passive_margin_geometry in passive_margin_geometries:
        passive_margin_feature = pygplates.Feature()
        passive_margin_feature.set_geometry(passive_margin_geometry)
        passive_margin_feature.set_valid_time(time + 0.5 * time_interval - 1e-4,  # epsilon to avoid overlap at interval boundaries
                                              time - 0.5 * time_interval)
        passive_margin_features.append(passive_margin_feature)

    # Save continent contours to GPML.
    pygplates.FeatureCollection(continent_contour_features).write(
        os.path.join(output_dir, 'continent_contour_features_{}.gpml'.format(time)))

    # Save passive margins to GPML.
    pygplates.FeatureCollection(passive_margin_features).write(
        os.path.join(output_dir, 'passive_margin_features_{}.gpml'.format(time)))

    # Save subducton zone segments to GPML (for debugging).
    pygplates.FeatureCollection(subduction_zone_features).write(
        os.path.join(output_dir, 'subduction_zone_features_{}.gpml'.format(time)))


def _find_passive_margin_geometries_on_contour(
        passive_margin_geometries,
        subduction_zone_lines,
        contoured_continent_segments):
    
    # Points for the current passive margin (if one).
    passive_margin_adjacent_points = []
    
    # Iterate over great circle arc segments of the contour.
    for contoured_continent_segment in contoured_continent_segments:
        # Create a polyline from the current contoured segment (so we can do distance testing).
        contoured_continent_line_segment = pygplates.PolylineOnSphere((
            contoured_continent_segment.get_start_point(),
            contoured_continent_segment.get_end_point()))
        
        # If continent segment near any subduction zone then it's an active margin.
        is_active_margin = False
        for subduction_zone_line in subduction_zone_lines:
            # If distance less than threshold distance.
            if pygplates.GeometryOnSphere.distance(
                    contoured_continent_line_segment, subduction_zone_line, max_distance_of_subduction_zone_from_active_margin_radians) is not None:
                is_active_margin = True
                break  # skip remaining subduction zones
        
        # If it's not an active margin then it's a passive margin.
        is_passive_margin = not is_active_margin
        
        # If segment a passive margin then add segment to current passive margin.
        if is_passive_margin:
            if not passive_margin_adjacent_points:
                # Add segment start point for first segment.
                passive_margin_adjacent_points.append(contoured_continent_line_segment[0])
            # Add segment end point.
            passive_margin_adjacent_points.append(contoured_continent_line_segment[1])
        else:  # active margin
            if passive_margin_adjacent_points:
                # We have accumulated passive margin points but are now in an active margin, so submit a passive margin.
                passive_margin_geometries.append(
                    pygplates.PolylineOnSphere(passive_margin_adjacent_points))
                # Clear points for next passive margin geometry
                del passive_margin_adjacent_points[:]
    
    # If there's one last passive margin geometry in the current contour then submit it.
    if passive_margin_adjacent_points:
        passive_margin_geometries.append(
            pygplates.PolylineOnSphere(passive_margin_adjacent_points))


if __name__ == '__main__':
    
    if use_all_cpus:
    
        # If 'use_all_cpus' is a bool (and therefore must be True) then use all available CPUs...
        if isinstance(use_all_cpus, bool):
            try:
                num_cpus = multiprocessing.cpu_count()
            except NotImplementedError:
                num_cpus = 1
        # else 'use_all_cpus' is a positive integer specifying the number of CPUs to use...
        elif isinstance(use_all_cpus, int) and use_all_cpus > 0:
            num_cpus = use_all_cpus
        else:
            raise TypeError('use_all_cpus: {} is neither a bool nor a positive integer'.format(use_all_cpus))
        
        # Distribute writing of each grid to a different CPU.
        with multiprocessing.Pool(num_cpus) as pool:
            pool.map(
                    partial(find_passive_margins),
                    times,
                    1) # chunksize
    
    else:
        for time in times:
            find_passive_margins(time)
    
    # Combine all features from each 'time'.
    continent_contour_features =  []
    passive_margin_features = []
    subduction_zone_features = []
    for time in times:
        continent_contour_filename = os.path.join(output_dir, 'continent_contour_features_{}.gpml'.format(time))
        continent_contour_features.extend(pygplates.FeatureCollection(continent_contour_filename))
        # Remove temporary file at current 'time'.
        if os.access(continent_contour_filename, os.R_OK):
            os.remove(continent_contour_filename)
        
        passive_margin_filename = os.path.join(output_dir, 'passive_margin_features_{}.gpml'.format(time))
        passive_margin_features.extend(pygplates.FeatureCollection(passive_margin_filename))
        # Remove temporary file at current 'time'.
        if os.access(passive_margin_filename, os.R_OK):
            os.remove(passive_margin_filename)
        
        subduction_zone_filename = os.path.join(output_dir, 'subduction_zone_features_{}.gpml'.format(time))
        subduction_zone_features.extend(pygplates.FeatureCollection(subduction_zone_filename))
        # Remove temporary file at current 'time'.
        if os.access(subduction_zone_filename, os.R_OK):
            os.remove(subduction_zone_filename)
    
    # Save ALL continent contours to GPMLZ.
    pygplates.FeatureCollection(continent_contour_features).write(
        os.path.join(output_dir, 'continent_contour_features.gpmlz'))
    
    # Save ALL passive margins to GPMLZ.
    pygplates.FeatureCollection(passive_margin_features).write(
        os.path.join(output_dir, 'passive_margin_features.gpmlz'))

    # Save ALL subducton zone segments to GPMLZ.
    pygplates.FeatureCollection(subduction_zone_features).write(
        os.path.join(output_dir, 'subduction_zone_features.gpmlz'))
