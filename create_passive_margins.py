from functools import partial
import math
import multiprocessing
import numpy as np
import os
import os.path
import pygplates
from ptt.continent_contours import ContinentContouring


#############################
# Start of input parameters #
#############################

# Location of Merdith et al (2021) 1Ga plate model.
model_dir = os.path.join('models', 'Merdith_etal_2021_Published')

# Rotation files (relative to input directory).
rotation_features = [pygplates.FeatureCollection(os.path.join(model_dir, file)) for file in (
    '1000_0_rotfile_Merdith_et_al.rot',
)]

# Topology features (absolute file paths).
#
# Only include those GPML files that are used for topologies.
topology_features = [pygplates.FeatureCollection(os.path.join(model_dir, file)) for file in (
    '250-0_plate_boundaries_Merdith_et_al.gpml',
    '410-250_plate_boundaries_Merdith_et_al.gpml',
    '1000-410-Convergence_Merdith_et_al.gpml',
    '1000-410-Divergence_Merdith_et_al.gpml',
    '1000-410-Topologies_Merdith_et_al.gpml',
    '1000-410-Transforms_Merdith_et_al.gpml',
    'TopologyBuildingBlocks_Merdith_et_al.gpml',
)]

# Continent polygon features (absolute file paths).
continent_features = [pygplates.FeatureCollection(os.path.join(model_dir, file)) for file in (
    'shapes_continents_Merdith_et_al.gpml',
)]

# Time range.
start_time = 0
end_time = 1000
time_interval = 5
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
max_distance_of_subduction_zone_from_active_margin_kms = 200
max_distance_of_subduction_zone_from_active_margin_radians = max_distance_of_subduction_zone_from_active_margin_kms / pygplates.Earth.mean_radius_in_kms

# The grid spacing (in degrees) between points in the grid used for contouring/aggregrating blocks of continental polygons.
continent_contouring_point_spacing_degrees = 1.0

# Contour polygons smaller than this will be excluded when contouring/aggregrating blocks of continental polygons.
#
# Note: Units here are for normalised sphere (ie, steradians or square radians) so full Earth area is 4*pi.
#       So 0.1 covers an area of approximately 4,000,000 km^2 (ie, 0.1 * 6371^2, where Earth radius is 6371km).
def continent_contouring_area_threshold_steradians(time):
    return 0.1

# Gaps between continent polygons smaller than this will be excluded when contouring/aggregrating blocks of continental polygons.
#
# Note: Units here are for normalised sphere (ie, radians).
#       So 1.0 radian is approximately 6371 km (where Earth radius is 6371 km).
#       Also 1.0 degree is approximately 110 km.
def continent_contouring_gap_threshold_radians(time):
    return math.radians(3.0)  # convert degrees to radians

###########################
# End of input parameters #
###########################



# Rotation model.
rotation_model = pygplates.RotationModel(rotation_features)

# Create a ContinentContouring object.
continent_contouring = ContinentContouring(
        rotation_model,
        continent_features,
        continent_contouring_point_spacing_degrees,
        continent_contouring_area_threshold_steradians,
        continent_contouring_gap_threshold_radians)
# Find passive margins at the specified time.
def find_passive_margins(time):
    print('time:', time)
    
    passive_margin_features = []
    contoured_features = []
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
            subduction_zone_feature.set_valid_time(time + 0.5 * time_interval, time - 0.5 * time_interval)
            subduction_zone_features.append(subduction_zone_feature)
    
    # Get the continent contours at the current time.
    contoured_continents = continent_contouring.get_contoured_continents(time)
    # Save contoured polygons as features (so we can later save them to a file for debugging).
    for contoured_continent in contoured_continents:
        contoured_feature = pygplates.Feature()
        contoured_feature.set_geometry(contoured_continent.get_polygons())
        contoured_feature.set_valid_time(time + 0.5 * time_interval, time - 0.5 * time_interval)
        contoured_features.append(contoured_feature)
    
    # Find passive margins along continent contours by removing active margins (contoured segments close to a subduction zone).
    passive_margin_geometries = []
    for contoured_continent in contoured_continents:
        for contoured_continent_polygon in contoured_continent.get_polygons():
            # Points for the current passive margin (if one).
            passive_margin_adjacent_points = []
            
            # Iterate over great circle arc segments of current contoured polygon.
            for contoured_continent_segment in contoured_continent_polygon.get_segments():
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
            
            # If there's one last passive margin geometry in the current contoured polygon then submit it.
            if passive_margin_adjacent_points:
                passive_margin_geometries.append(
                    pygplates.PolylineOnSphere(passive_margin_adjacent_points))
    
    # Convert any passive margin geometries found into features.
    for passive_margin_geometry in passive_margin_geometries:
        passive_margin_feature = pygplates.Feature()
        passive_margin_feature.set_geometry(passive_margin_geometry)
        passive_margin_feature.set_valid_time(time + 0.5 * time_interval, time - 0.5 * time_interval)
        passive_margin_features.append(passive_margin_feature)

    # Save passive margins to GPML.
    pygplates.FeatureCollection(passive_margin_features).write('passive_margin_features_{}.gpml'.format(time))

    # Save contoured continents to GPML (for debugging).
    pygplates.FeatureCollection(contoured_features).write('contoured_features_{}.gpml'.format(time))

    # Save subducton zone segments to GPML (for debugging).
    pygplates.FeatureCollection(subduction_zone_features).write('subduction_zone_features_{}.gpml'.format(time))


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
    passive_margin_features = []
    contoured_features = []
    subduction_zone_features = []
    for time in times:
        passive_margin_filename = 'passive_margin_features_{}.gpml'.format(time)
        passive_margin_features.extend(pygplates.FeatureCollection(passive_margin_filename))
        # Remove temporary file at current 'time'.
        if os.access(passive_margin_filename, os.R_OK):
            os.remove(passive_margin_filename)
        
        contoured_filename = 'contoured_features_{}.gpml'.format(time)
        contoured_features.extend(pygplates.FeatureCollection(contoured_filename))
        # Remove temporary file at current 'time'.
        if os.access(contoured_filename, os.R_OK):
            os.remove(contoured_filename)
        
        subduction_zone_filename = 'subduction_zone_features_{}.gpml'.format(time)
        subduction_zone_features.extend(pygplates.FeatureCollection(subduction_zone_filename))
        # Remove temporary file at current 'time'.
        if os.access(subduction_zone_filename, os.R_OK):
            os.remove(subduction_zone_filename)
    
    # Save ALL passive margins to GPML.
    pygplates.FeatureCollection(passive_margin_features).write('passive_margin_features.gpml')

    # Save ALL contoured continents to GPML (for debugging).
    pygplates.FeatureCollection(contoured_features).write('contoured_features.gpml')

    # Save ALL subducton zone segments to GPML (for debugging).
    pygplates.FeatureCollection(subduction_zone_features).write('subduction_zone_features.gpml')
