from __future__ import annotations

import os
import time
from math import asin, cos, radians, sin, sqrt

import folium
import osmnx as ox
import streamlit as st
from geopy.geocoders import Nominatim
from pyproj import Transformer
from streamlit_folium import st_folium

from graph_utils import (
    build_weight_matrices,
    default_comfort,
    exact_route,
    greedy_route,
    load_road_graph,
    point_distances,
    restore_route,
)


APP_TITLE = "Оптимальный объезд точек"
COUNTRY_CODES = "ru"
DEFAULT_CENTER = [55.751244, 37.618423]
SEARCH_LIMIT = 5


def init_session_state():
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "points" not in st.session_state:
        st.session_state.points = []
    if "route" not in st.session_state:
        st.session_state.route = None
    if "coefficients" not in st.session_state:
        st.session_state.coefficients = default_comfort.copy()


def clear_route():
    st.session_state.route = None


@st.cache_resource
def get_geolocator():
    return Nominatim(user_agent="student_route_planner_streamlit")


@st.cache_data(ttl=3600, show_spinner=False)
def search_address(query):
    time.sleep(1.1)
    locations = get_geolocator().geocode(
        query,
        exactly_one=False,
        limit=SEARCH_LIMIT,
        country_codes=COUNTRY_CODES,
        language="ru",
        timeout=10,
    ) or []
    return [
        {"address": location.address, "lat": location.latitude,
         "lon": location.longitude}
        for location in locations
    ]


def distance_between(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    value = sin((lat2 - lat1) / 2) ** 2
    value += cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 12742000 * asin(sqrt(value))


@st.cache_resource
def load_graph(center, radius):
    ox.settings.use_cache = True
    return load_road_graph(center, radius)


def build_route():
    points = st.session_state.points
    center = (points[0]["lat"], points[0]["lon"])
    radius = max(
        distance_between(center[0], center[1], point["lat"], point["lon"])
        for point in points
    ) + 1000
    graph, projected, node_to_idx, idx_to_node = load_graph(center, radius)
    matrices, node_to_idx, idx_to_node = build_weight_matrices(
        projected, st.session_state.coefficients
    )
    transformer = Transformer.from_crs(
        "EPSG:4326", projected.graph["crs"], always_xy=True
    )
    point_indices = []
    for point in points:
        x, y = transformer.transform(point["lon"], point["lat"])
        node = ox.distance.nearest_nodes(projected, x, y)
        point_indices.append(node_to_idx[node])

    distances, predecessors = point_distances(
        matrices[st.session_state.criterion], point_indices
    )
    if len(point_indices) <= 10:
        order, score = exact_route(distances)
    else:
        order, score = greedy_route(distances)
    route_nodes = restore_route(order, point_indices, predecessors, idx_to_node)
    total_distance = 0
    total_time = 0
    for start, end in zip(route_nodes[:-1], route_nodes[1:]):
        start_index = node_to_idx[start]
        end_index = node_to_idx[end]
        total_distance += matrices["distance"][start_index, end_index]
        total_time += matrices["time"][start_index, end_index]
    st.session_state.route = {
        "graph": graph,
        "nodes": route_nodes,
        "order": order,
        "distance": total_distance,
        "time": total_time,
        "score": score,
    }


def create_map(points, route):
    if points:
        center = [points[0]["lat"], points[0]["lon"]]
    else:
        center = DEFAULT_CENTER
    tiles = "https://basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png"
    key = os.getenv("CARTO_BASEMAPS_API_KEY", "")
    if key:
        tiles += f"?key={key}"
    map_object = folium.Map(
        location=center, zoom_start=12, tiles=tiles,
        attr="© OpenStreetMap contributors, © CARTO", control_scale=True
    )
    for index, point in enumerate(points):
        color = "red" if index == 0 else "blue"
        folium.Marker(
            [point["lat"], point["lon"]],
            tooltip=point["address"],
            icon=folium.Icon(color=color),
        ).add_to(map_object)
    if route is not None:
        line = [
            [route["graph"].nodes[node]["y"], route["graph"].nodes[node]["x"]]
            for node in route["nodes"]
        ]
        folium.PolyLine(line, color="green", weight=5).add_to(map_object)
    return map_object


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()
    st.title(APP_TITLE)

    controls, map_column = st.columns([0.38, 0.62], gap="large")
    with controls:
        st.subheader("Поиск адреса")
        with st.form("address_search_form"):
            query = st.text_input("Адрес")
            search_submitted = st.form_submit_button("Найти")
        if search_submitted and query.strip():
            st.session_state.search_results = search_address(query.strip())
        if st.session_state.search_results:
            selected_index = st.selectbox(
                "Результаты поиска",
                range(len(st.session_state.search_results)),
                format_func=lambda index: st.session_state.search_results[index]["address"],
            )
            if st.button("Добавить точку"):
                st.session_state.points.append(
                    st.session_state.search_results[selected_index].copy()
                )
                st.session_state.route = None

        st.subheader("Точки маршрута")
        for index, point in enumerate(st.session_state.points):
            name_column, button_column = st.columns([4, 1])
            name_column.write(f"{index + 1}. {point['address']}")
            if button_column.button("Удалить", key=f"remove_{index}"):
                st.session_state.points.pop(index)
                st.session_state.route = None
                st.rerun()
        if st.button("Очистить всё"):
            st.session_state.points = []
            st.session_state.route = None

        st.selectbox(
            "Критерий",
            ["distance", "time", "comfort"],
            format_func=lambda value: {
                "distance": "Расстояние",
                "time": "Время",
                "comfort": "Комфорт",
            }[value],
            key="criterion",
            on_change=clear_route,
        )
        if st.session_state.criterion == "comfort":
            with st.form("comfort_form"):
                coefficients = {}
                for road_type, value in st.session_state.coefficients.items():
                    coefficients[road_type] = st.number_input(
                        road_type, min_value=0.1,
                        value=float(value), step=0.1,
                    )
                if st.form_submit_button("Применить коэффициенты"):
                    st.session_state.coefficients = coefficients
                    st.session_state.route = None

        if st.button("Построить маршрут") and len(st.session_state.points) >= 2:
            with st.spinner("Загружаю дорожную сеть и строю маршрут"):
                build_route()

    with map_column:
        st.subheader("Карта")
        map_object = create_map(st.session_state.points, st.session_state.route)
        st_folium(
            map_object, height=560, use_container_width=True,
            returned_objects=[],
        )
        if st.session_state.route is not None:
            route = st.session_state.route
            st.write(f"Длина маршрута: {route['distance'] / 1000:.2f} км")
            st.write(f"Время в пути: {route['time'] / 60:.1f} мин")
            st.write("Порядок точек:")
            for point_index in route["order"]:
                st.write(st.session_state.points[point_index]["address"])


if __name__ == "__main__":
    main()
