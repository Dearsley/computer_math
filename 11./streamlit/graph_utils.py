import osmnx as ox

def load_road_graph(center, radius):
    graph = ox.graph_from_point(center, dist=radius, network_type="drive")
    graph_utm = ox.project_graph(graph)
    nodes = list(graph_utm.nodes)
    node_to_idx = {node: index for index, node in enumerate(nodes)}
    idx_to_node = {index: node for index, node in enumerate(nodes)}
    return graph, graph_utm, node_to_idx, idx_to_node

import re
from scipy.sparse import lil_matrix

default_speeds = {
    "motorway": 110, "trunk": 90, "primary": 60,
    "secondary": 50, "tertiary": 40, "residential": 30,
    "living_street": 20, "service": 20, "unclassified": 30,
}
default_comfort = {
    "motorway": 0.7, "trunk": 0.8, "primary": 0.85,
    "secondary": 0.9, "tertiary": 1.0, "residential": 1.2,
    "living_street": 1.4, "service": 1.5, "unclassified": 1.3,
}

def build_weight_matrices(graph, comfort_coefficients=None):
    nodes = list(graph.nodes)
    node_to_idx = {node: index for index, node in enumerate(nodes)}
    idx_to_node = {index: node for index, node in enumerate(nodes)}
    count = len(nodes)
    distances = lil_matrix((count, count), dtype=float)
    times = lil_matrix((count, count), dtype=float)
    comfort = lil_matrix((count, count), dtype=float)
    comfort_values = default_comfort.copy()
    if comfort_coefficients is not None:
        comfort_values.update(comfort_coefficients)

    for start, end, data in graph.edges(data=True):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]
        speed = data.get("maxspeed", default_speeds.get(highway, 30))
        if isinstance(speed, list):
            speed = speed[0]
        speed_numbers = re.findall(r"\d+(?:\.\d+)?", str(speed))
        speed_kmh = float(speed_numbers[0]) if speed_numbers else default_speeds.get(highway, 30)
        length = data["length"]
        i = node_to_idx[start]
        j = node_to_idx[end]
        weights = [
            (distances, length),
            (times, length / (speed_kmh / 3.6)),
            (comfort, length * comfort_values.get(highway, 1.3)),
        ]
        for matrix, weight in weights:
            if matrix[i, j] == 0 or weight < matrix[i, j]:
                matrix[i, j] = weight

    matrices = {
        "distance": distances.tocsr(),
        "time": times.tocsr(),
        "comfort": comfort.tocsr(),
    }
    return matrices, node_to_idx, idx_to_node

from scipy.sparse.csgraph import dijkstra

def point_distances(matrix, point_indices):
    distances, predecessors = dijkstra(
        csgraph=matrix,
        directed=True,
        indices=point_indices,
        return_predecessors=True,
    )
    between_points = distances[:, point_indices]
    return between_points, predecessors

from itertools import permutations

def exact_route(distances, depot=0):
    points = [i for i in range(len(distances)) if i != depot]
    best_length = float("inf")
    best_route = []
    for order in permutations(points):
        route = [depot] + list(order) + [depot]
        length = 0
        for i in range(len(route) - 1):
            length += distances[route[i], route[i + 1]]
        if length < best_length:
            best_length = length
            best_route = route
    return best_route, best_length

def greedy_route(distances, depot=0):
    remaining = set(range(len(distances)))
    remaining.remove(depot)
    route = [depot]
    length = 0
    current = depot
    while remaining:
        next_point = min(remaining, key=lambda point: distances[current, point])
        length += distances[current, next_point]
        route.append(next_point)
        remaining.remove(next_point)
        current = next_point
    length += distances[current, depot]
    route.append(depot)
    return route, length

def restore_route(order, point_indices, predecessors, idx_to_node):
    route_indices = []
    for start_point, end_point in zip(order[:-1], order[1:]):
        start = point_indices[start_point]
        current = point_indices[end_point]
        segment = [current]
        while current != start:
            current = int(predecessors[start_point, current])
            segment.append(current)
        segment.reverse()
        if route_indices:
            segment = segment[1:]
        route_indices.extend(segment)
    return [idx_to_node[index] for index in route_indices]

def recalculate_comfort(graph, coefficients):
    matrices, node_to_idx, idx_to_node = build_weight_matrices(
        graph, coefficients
    )
    return matrices["comfort"]
