"""
Graph analysis utilities for synapse detection.
"""
import skan
import itertools
import numpy as np
import networkx as nx
import skimage as ski
import matplotlib.pyplot as plt
from skimage.draw import line
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist


def get_synapses_graph(worm_mask: np.ndarray,
                      maxima_coords: np.ndarray,
                      n_segments: int = 20) -> Tuple[np.ndarray, nx.Graph, float, float, float, np.ndarray, np.ndarray]:
    """
    Create graph representation of synapses.
    
    Args:
        worm_mask: Binary mask of worm
        maxima_coords: Coordinates of maxima points
        n_segments: Number of segments to divide worm into
        
    Returns:
        Tuple of (maxima coordinates, graph, median width, slice difference measure,
                point segment difference measure, mask head, mask queue)
    """
    # Default return value for error cases
    DEFAULT_RETURN = (np.array([]), nx.Graph(), 0.0, 0.0, 0.0, np.array([]), np.array([]))
    
    # ----- Input validation -----
    if worm_mask is None or worm_mask.size == 0 or maxima_coords is None or len(maxima_coords) == 0:
        print("Warning: Empty worm mask or maxima coordinates provided")
        return DEFAULT_RETURN
        
    # -- 1 -- Skeletonize worm and get main branch
    skeleton = ski.morphology.skeletonize(worm_mask)
    G = _skeleton_to_graph(skeleton)  # Create initial graph from skeleton
    G, skeleton = _skeleton_keep_main_branch(G, skeleton, maxima_coords, keep=1)
    if len(G.nodes) == 0: print("Warning: No nodes found in skeleton"); return DEFAULT_RETURN

    # -- 2 -- Decompose skeleton into segments
    skel_path = _order_skeleton_points_skan(skeleton)
    if not skel_path: print("Warning: No skeleton points found"); return DEFAULT_RETURN
    n = len(skel_path)
    seg_len = max(1, n // n_segments)  # Ensure non-zero segment length
    centers = [skel_path[i * seg_len + seg_len // 2] for i in range(min(n_segments, n))]
    if not centers: print("Warning: No segment centers found"); return DEFAULT_RETURN

    # -- 3 -- Calculate segment statistics
    distances = cdist(maxima_coords, centers)
    labels = np.argmin(distances, axis=1)
    counts = np.bincount(labels, minlength=len(centers))
    if len(counts) == 0: print("Warning: No valid segment counts"); return DEFAULT_RETURN
    mean_count = np.mean(counts)
    diff_points = counts - mean_count
    measure_diff_points = np.sqrt(np.sum(diff_points**2))

    # -- 4 -- Calculate segment directions
    directions = []
    for i in range(min(n_segments, n)):
        start_idx = i * seg_len
        end_idx = min((i + 1) * seg_len - 1, n - 1)
                
        if start_idx >= n or end_idx >= n: continue
                    
        start = np.array(skel_path[start_idx])
        end = np.array(skel_path[end_idx])
        vec = end - start
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 0 else np.zeros_like(start, dtype=float)
        directions.append(vec)
    if not directions: print("Warning: No valid segment directions found"); return DEFAULT_RETURN

    # -- 5 -- Decompose segments into slices
    # Set entire border to black
    worm_mask[0, :] = worm_mask[-1, :] = worm_mask[:, 0] = worm_mask[:, -1] = 0
    dic_segments, median_width = decompose_worm_segments_into_slice(skel_path, worm_mask, n_segments)
        
    # Assign each maxima to the slice it belongs to
    labels_slice = np.zeros(len(maxima_coords), dtype=int)
    def get_slice_label(dist, threshold1, threshold2):
        """Determine slice label based on distance and thresholds."""
        if dist < threshold1 and dist < threshold2:
            return 0  # Both thresholds exceeded (closest)
        elif dist > threshold1 and dist < threshold2:
            return 1  # Middle range
        else:  # dist > threshold1 and dist > threshold2
            return 2  # Furthest range

    for i, point in enumerate(maxima_coords):
        seg_idx = labels[i]
        segment = skel_path[seg_idx * seg_len: (seg_idx + 1) * seg_len]
        
        dist_to_seg, sign = _signed_distance_to_segment_2d(segment, point)
        
        if sign > 0:
            # Positive direction: use indices 5 and 7
            slice_offset = get_slice_label(dist_to_seg, dic_segments[seg_idx][5], dic_segments[seg_idx][7])
            labels_slice[i] = 2 - slice_offset  # Maps 0,1,2 to 2,1,0
        else:
            # Negative direction: use indices 6 and 8  
            slice_offset = get_slice_label(dist_to_seg, dic_segments[seg_idx][6], dic_segments[seg_idx][8])
            labels_slice[i] = 3 + slice_offset  # Maps 0,1,2 to 3,4,5     

    # IMAGE_DIAPO
    """plt.figure(figsize=(8, 8))
    plt.imshow(worm_mask, cmap='gray')
    for i in range(n_segments-1):
        plt.plot([dic_segments[i][4][1], dic_segments[i][3][1]], [dic_segments[i][4][0], dic_segments[i][3][0]], 'g-', label=f'Segment {i}' if i == 0 else "")
        start = dic_segments[i][0]
        end = dic_segments[i+1][0]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
        start = dic_segments[i][1]
        end = dic_segments[i+1][1]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
        start = dic_segments[i][2]
        end = dic_segments[i+1][2]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
        start = dic_segments[i][3]
        end = dic_segments[i+1][3]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
        start = dic_segments[i][4]
        end = dic_segments[i+1][4]
        plt.plot([start[1], end[1]], [start[0], end[0]], 'b--', label=f'Segment {i}' if i == 0 else "")
    plt.plot([dic_segments[n_segments-1][4][1], dic_segments[n_segments-1][3][1]], [dic_segments[n_segments-1][4][0], dic_segments[n_segments-1][3][0]], 'g-', label=f'Segment {n_segments-1}' if n_segments-1 == 0 else "")
    plt.title("Segment Directions with Perpendicular Lines (Both Directions)")
    plt.show()"""

    # -- 6 -- Calculate segment statistics
    Nb_slice = np.zeros(6, dtype=int)
    Diff_slice = np.zeros(6, dtype=int)
    for i in range(6):
        Nb_slice[i] = np.sum(labels_slice == i)
        Diff_slice[i] = Nb_slice[i] - np.mean(Nb_slice)
    measure_diff_slice = np.sqrt(np.sum(Diff_slice**2))

    # -- 7 -- Create graph from maxima coordinates and segment directions
    G = nx.Graph()
    for i, p1 in enumerate(maxima_coords):
        node1 = tuple(p1)
        G.add_node(node1, centroid=p1)
        seg_idx1 = min(labels[i], len(directions) - 1)
        dir_vec = directions[seg_idx1]
        slices1 = labels_slice[i]

        best_j = None
        min_dist = np.inf

        for j, p2 in enumerate(maxima_coords):
            if i == j: continue
                    
            if abs(slices1 - labels_slice[j]) > 1: continue
                    
            vec = p2 - p1
            dist = np.linalg.norm(vec)
                    
            if dist == 0: continue
                    
            vec_normed = vec / dist
            dot = np.dot(vec_normed, dir_vec)
            
            if dot > 0.9 and dist < min_dist:
                best_j = j
                min_dist = dist

        if best_j is not None:
            node2 = tuple(maxima_coords[best_j])
            G.add_node(node2, centroid=maxima_coords[best_j])
            G.add_edge(node1, node2)


    # -- 8 -- Final processing of graph and skeleton
    if np.sum(np.isin(labels_slice, [0, 1])) > len(maxima_coords) / 5 and np.sum(np.isin(labels_slice, [4, 5])) > len(maxima_coords) / 5:
        NUMBER_OF_CORDS = 2
    else:
        NUMBER_OF_CORDS = 1
    skeleton = _graph_to_skeleton(G, shape=worm_mask.shape)
      
    G, skeleton = _skeleton_keep_main_branch(G, skeleton, maxima_coords, keep=NUMBER_OF_CORDS)

    # Get final maxima
    maxima = np.array([node for node in maxima_coords if 0 <= node[0] < skeleton.shape[0] 
                      and 0 <= node[1] < skeleton.shape[1] and skeleton[node[0], node[1]] == 1])                  
    if len(maxima) == 0: print("Warning: No valid maxima points found in skeleton"); return DEFAULT_RETURN
                
    # -- 9 -- Get the nerve ring

    # Find the head - compare width
    head_is_first = dic_segments[0][9] > dic_segments[n_segments-1][9]

    # Find the head - compare intensity - create two masks : one for each extremity
    head_mask_1 = np.zeros_like(worm_mask)
    head_mask_2 = np.zeros_like(worm_mask)
    mask_coords = np.column_stack(np.where(worm_mask))  # Shape: (N, 2) as (row, col)
    head_center_1 = np.array(dic_segments[0][0])  # (row, col)
    head_center_2 = np.array(dic_segments[n_segments-1][0])  # (row, col)
    distances_1 = np.linalg.norm(mask_coords - head_center_1, axis=1) # Calculate distance
    distances_2 = np.linalg.norm(mask_coords - head_center_2, axis=1)
    threshold = 150
    close_to_head_1 = distances_1 <= threshold
    close_to_head_2 = distances_2 <= threshold
    head_mask_1[mask_coords[close_to_head_1, 0], mask_coords[close_to_head_1, 1]] = 1
    head_mask_2[mask_coords[close_to_head_2, 0], mask_coords[close_to_head_2, 1]] = 1
 

    # IMAGE_DIAPO
    """plt.figure(figsize=(12, 8))
    plt.imshow(worm_mask, cmap='gray', alpha=0.8)
    plt.imshow(np.ma.masked_where(head_mask_1 == 0, head_mask_1), 
            cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    plt.imshow(np.ma.masked_where(head_mask_2 == 0, head_mask_2), 
            cmap='Greens', alpha=0.6, vmin=0, vmax=1)
    if head_is_first:
        head_point = dic_segments[0][0]
        plt.plot(head_point[1], head_point[0], 'ro', markersize=8, label='Head (First segment)')
    else:
        head_point = dic_segments[n_segments-1][0]
        plt.plot(head_point[1], head_point[0], 'ro', markersize=8, label='Head (Last segment)')
    plt.title('Worm Head Detection Analysis', fontsize=14)
    plt.legend()
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Head Point'),
        Patch(facecolor='blue', alpha=0.6, label='Head Mask 1 (First segment)'),
        Patch(facecolor='green', alpha=0.6, label='Head Mask 2 (Last segment)'),
        Patch(facecolor='white', alpha=0.8, label='Worm Mask')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.axis('off')  # Remove axes for cleaner look
    plt.tight_layout()
    plt.show()"""
    
    return maxima, G, median_width, measure_diff_slice, measure_diff_points, head_mask_1, head_mask_2

# Utils functions
def decompose_worm_segments_into_slice(skel_path, worm_mask, n_segments):
    """Analyze worm segments and return dic_segments with visualization."""
    seg_len = len(skel_path) // n_segments
    
    # Find coordinate of the middle of each segment
    middle_coords = [skel_path[i * seg_len + seg_len // 2] for i in range(n_segments)]
    
    dic_segments = {}
    
    for i in range(n_segments):
        start = middle_coords[i]
        end = skel_path[min((i + 1) * seg_len - 1, len(skel_path) - 1)]
        
        # Calculate direction vector and perpendicular
        direction_vec = np.array(end) - np.array(start)
        if np.linalg.norm(direction_vec) > 0:
            direction_vec = direction_vec / np.linalg.norm(direction_vec)
        
        perp_vec = np.array([-direction_vec[1], direction_vec[0]])
        
        # Find intersections in both directions
        end_pos = _find_mask_intersection(start, perp_vec, worm_mask)
        end_neg = _find_mask_intersection(start, -perp_vec, worm_mask)
        
        # Calculate all segment properties and store in original format
        dic_segments[i] = _calculate_segment_properties(start, end_pos, end_neg)
    
    # Calculate median width
    median_width = np.median([dic_segments[i][9] for i in range(n_segments)])
    
    return dic_segments, median_width

def _clamp_coordinates(coords, max_val=1024):
    """Clamp coordinates to valid image bounds."""
    return tuple(np.clip(coords, 0, max_val))

def _find_mask_intersection(start, direction, worm_mask, max_distance=1000):
    """Find intersection point with mask boundary along a direction."""
    end_point = start + direction * max_distance
    end_point = np.round(end_point).astype(int)
    
    # Get line coordinates
    rr, cc = line(start[0], start[1], end_point[0], end_point[1])
    
    # Find first intersection with mask boundary (value == 0)
    for r, c in zip(rr, cc):
        if 0 <= r < worm_mask.shape[0] and 0 <= c < worm_mask.shape[1]:
            if worm_mask[r, c] == 0:
                return _clamp_coordinates((r, c), max(worm_mask.shape) - 1)
    
    return _clamp_coordinates(end_point, max(worm_mask.shape) - 1)

def _calculate_segment_properties(start, end_pos, end_neg):
    """Calculate all segment properties and return as tuple matching original format."""
    mid_pos = ((start[0] + end_pos[0]) // 2, (start[1] + end_pos[1]) // 2)
    mid_neg = ((start[0] + end_neg[0]) // 2, (start[1] + end_neg[1]) // 2)
    
    length_mid_pos = np.linalg.norm(np.array(mid_pos) - np.array(start))
    length_mid_neg = np.linalg.norm(np.array(mid_neg) - np.array(start))
    length_end_pos = np.linalg.norm(np.array(end_pos) - np.array(start))
    length_end_neg = np.linalg.norm(np.array(end_neg) - np.array(start))
    length_total = np.linalg.norm(np.array(end_pos) - np.array(end_neg))
    
    return (start, mid_pos, mid_neg, end_pos, end_neg, 
            length_mid_pos, length_mid_neg, length_end_pos, length_end_neg, length_total)

def _find_endpoints_graph(G: 'nx.Graph', 
                  maxima_coords: np.ndarray,
                  angle_threshold_degrees: float = 90) -> Tuple[List, List]:
    """
    Find endpoints and angle junctions in a graph.
    
    Args:
        G: NetworkX graph
        maxima_coords: Coordinates of maxima points
        angle_threshold_degrees: Threshold for angle detection
        
    Returns:
        Tuple of (endpoints, angle junctions) as lists of coordinate tuples
    """
    
    # Input validation
    if G is None or len(G.nodes) == 0 or maxima_coords is None or len(maxima_coords) == 0:
        print("Warning: Empty graph or maxima coordinates provided to find_endpoints")
        return [], []
    
    # Find endpoints (nodes with degree 1)
    endpoints = []
    for node in G.nodes:
        if G.degree[node] == 1:
            if isinstance(node, tuple):
                endpoints.append(node)
            elif isinstance(node, (int, np.integer)):
                if node < len(maxima_coords):
                    endpoints.append(tuple(maxima_coords[node]))       
    if not endpoints: print("Warning: No endpoints found"); return [], []
    
    # Process each node for finding angle junctions
    angle_junctions = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        
        if len(neighbors) >= 2:
            coords = []
            # Gérer le cas où les nœuds sont déjà des coordonnées
            if isinstance(node, tuple):
                node_coords = np.array(node)
                coords = [np.array(n) for n in neighbors]
            # Gérer le cas où les nœuds sont des indices
            elif isinstance(node, (int, np.integer)) and node < len(maxima_coords):
                node_coords = maxima_coords[node]
                coords = [maxima_coords[n] for n in neighbors if isinstance(n, (int, np.integer)) and n < len(maxima_coords)]
            else:
                continue
                
            if not coords:
                continue
            
            all_angles_sharp = True
            
            for a, b in itertools.combinations(coords, 2):
                v1 = a - node_coords
                v2 = b - node_coords
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 == 0 or norm2 == 0:
                    continue
                
                cosine_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle_rad = np.arccos(cosine_angle)
                angle_deg = np.degrees(angle_rad)
                
                if angle_deg >= angle_threshold_degrees:
                    all_angles_sharp = False
                    break
            
            if all_angles_sharp:
                angle_junctions.append(tuple(node_coords))
    
    return endpoints, angle_junctions

def _skeleton_to_graph(skel: np.ndarray) -> 'nx.Graph':
    """
    Convert skeleton image to graph.
    
    Args:
        skel: Skeleton image
        
    Returns:
        NetworkX graph
    """
        
    # Input validation
    if skel is None or skel.size == 0: print("Warning: Empty skeleton provided to skeleton_to_graph"); return nx.Graph()
            
    G = nx.Graph()
        
    # Find non-zero coordinates
    coords = np.column_stack(np.where(skel)) # extract coordinates of non-zero pixels. coords is a 2D array of shape (n, 2) where n is the number of non-zero pixels.
    if len(coords) == 0: print("Warning: No non-zero coordinates found in skeleton"); return nx.Graph()

    # Add edges between neighboring pixels
    for y, x in coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                    if skel[ny, nx]:
                        G.add_edge((y, x), (ny, nx))
                            
    if len(G.nodes) == 0: print("Warning: No nodes added to graph")
    
    return G

def _graph_to_skeleton(G: 'nx.Graph',
                     shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Convert graph to skeleton image.
    
    Args:
        G: NetworkX graph
        shape: Output image shape
        
    Returns:
        Binary skeleton image
    """
    # Input validation
    if G is None or len(G.nodes) == 0 : 
        print("Warning: Empty graph provided to graph_to_skeleton"); 
        return np.zeros((1, 1), dtype=bool) if shape is None else np.zeros(shape, dtype=bool)
     
    skel = np.zeros(shape, dtype=bool)
    
    # Set node points
    for node in G.nodes:        
        y, x = node
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            skel[y, x] = True
        else: print(f"Warning: Coordinates ({y}, {x}) out of bounds for shape {shape}")
            
    # Draw edges
    for edge in G.edges:
        y1, x1 = edge[0]
        y2, x2 = edge[1]
            
        if (0 <= y1 < shape[0] and 0 <= x1 < shape[1] and 
            0 <= y2 < shape[0] and 0 <= x2 < shape[1]):
            rr, cc = line(y1, x1, y2, x2)
            skel[rr, cc] = True
            
    return skel

def _order_skeleton_points_skan(skeleton):
    # Create the Skeleton object
    skel_obj = skan.csr.Skeleton(skeleton)
    
    # Get the summary with branch information
    try:
        summary = skan.summarize(skel_obj) 
    except:
        summary = skan.summarize(skel_obj, separator='-') 
    
    # Create a flat list of all points from all paths
    all_points = []
    
    for i in range(len(summary)):
        # Get coordinates for each path
        path_coords = skel_obj.path_coordinates(i)
        # Add all points from this path to the flat list
        for coord in path_coords:
            all_points.append(tuple(coord))
    
    return all_points

def _skeleton_keep_main_branch(G: 'nx.Graph',
                            skel: np.ndarray,
                            maxima_coords: np.ndarray,
                            keep: int = 1) -> Tuple['nx.Graph', np.ndarray]:
    """
    Keep only the main branches of a skeleton.
    
    Args:
        G: Input graph
        skel: Skeleton image
        maxima_coords: Coordinates of maxima points
        keep: Number of branches to keep
        
    Returns:
        Tuple of (processed graph, main branch mask)
    """
    # Input validation
    if (skel is None or skel.size == 0 or 
        maxima_coords is None or len(maxima_coords) == 0 or 
        len(G.nodes) == 0):
        return nx.Graph(), np.zeros((1, 1), dtype=bool)
    
    # Find endpoints and junctions
    endpoints, angle_junctions = _find_endpoints_graph(G, maxima_coords)
    if len(endpoints) < 2:
        return G, skel  # Return original if insufficient endpoints

    # Find all paths between endpoints and calculate metrics
    all_paths = []
    angle_junction_set = set(angle_junctions)
    
    for i, j in itertools.combinations(range(len(endpoints)), 2):
        try:
            path = nx.shortest_path(G, endpoints[i], endpoints[j])

            if path:
                # Split path at junctions if they exist
                subpaths = _split_path_at_junctions(path, angle_junction_set)
                all_paths.extend(subpaths)
        except nx.NetworkXNoPath:
            continue

    if not all_paths:
        return G, np.zeros_like(skel, dtype=bool)
    
    # Select best paths based on maxima count
    selected_paths = _select_best_paths(all_paths, keep)
    
    # Create binary mask for selected paths
    main_branch = np.zeros(skel.shape, dtype=bool)
    for path in selected_paths:
        for node in path:
            x,y = node
            main_branch[x,y] = True

    return G, main_branch

def _split_path_at_junctions(path, angle_junction_set):
    """Split a path at junction points and return list of subpaths with metrics."""
    
    # Find junction indices in the path
    junction_indices = []
    for i, node in enumerate(path):
        if (node in angle_junction_set):
            junction_indices.append(i)
    
    if len(junction_indices) == 0:
        return [{
                "path": path,
                "maxima_count": len(path)
                }]
    else:
        # Create split points (start, junctions, end)
        split_points = [0] + junction_indices + [len(path) - 1]
        split_points = sorted(set(split_points))
        
        subpaths = []
        for start, end in zip(split_points[:-1], split_points[1:]):
            if start < end:
                subpath = path[start:end + 1]
                subpaths.append({
                    "path": subpath,
                    "maxima_count": len(subpath)
                })
                
        return subpaths

def _select_best_paths(all_paths, keep):
    """Select the best paths based on maxima count, avoiding overlap."""
    sorted_paths = sorted(all_paths, key=lambda x: x["maxima_count"], reverse=True)
    
    selected_paths = []
    used_nodes = set()
    
    for path_data in sorted_paths:
        path = path_data["path"]
        path_nodes = set(path)
        
        # For keep=1, just take the first (best) path
        # For keep=2, avoid overlapping paths
        if ((keep == 1 and not selected_paths) or 
            (keep == 2 and not used_nodes.intersection(path_nodes))):
            selected_paths.append(path)
            used_nodes.update(path_nodes)
            
            if len(selected_paths) == keep:
                break
    
    return selected_paths

def _signed_distance_to_segment_2d(seg, p1):
    seg = np.array(seg)  # Convert 'seg' to a NumPy array
    if seg.shape[0] < 2:
        # Handle cases where 'seg' doesn't define a segment
        if seg.shape[0] == 1:
            return np.linalg.norm(seg[0] - np.array(p1)), 0
        else:
            return np.inf, 0
    v = seg[-1] - seg[0]
    w = seg[0] - np.array(p1) # Ensure p1 is also a NumPy array for subtraction
    cross_product = v[0] * w[1] - v[1] * w[0]
    segment_length = np.linalg.norm(v)
    if segment_length == 0:
        return np.linalg.norm(w), 0
    distance = np.abs(cross_product) / segment_length
    return distance, np.sign(cross_product)

    """Create a binary mask for the selected paths."""
    main_branch = np.zeros(shape, dtype=bool)

    print(f"Shape: {shape}")

    
    for path in selected_paths:
        for node in path:
            x, y = maxima_coords[node]
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                main_branch[y, x] = True
    
    return main_branch