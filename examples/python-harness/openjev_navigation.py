"""Read Quake II AAS reachabilities for the external UDP companion."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import struct


# Layouts from src/botlib/aas/aas_map.h and aas_local.h.
HEADER = struct.Struct("<ii")
LUMP = struct.Struct("<ii")
AREA = struct.Struct("<iii9f")
SETTINGS = struct.Struct("<7i")
REACH = struct.Struct("<iii6fiHH")
AAS_IDENT = int.from_bytes(b"EAAS", "little")
WALKABLE = {2, 3, 4, 5, 6, 7, 8, 9}
JUMP_TYPES = {4, 5, 9}


@dataclass(frozen=True)
class Area:
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]
    center: tuple[float, float, float]


@dataclass(frozen=True)
class Edge:
    destination: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    travel_type: int
    travel_time: int


def horizontal_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class AASNavigator:
    def __init__(self, path: Path):
        data = path.read_bytes()
        ident, version = HEADER.unpack_from(data)
        if ident != AAS_IDENT or version not in (2, 3):
            raise ValueError(f"unsupported AAS header in {path}")
        lumps = [LUMP.unpack_from(data, HEADER.size + index * LUMP.size)
                 for index in range(14)]

        def rows(index: int, layout: struct.Struct) -> list[tuple]:
            offset, length = lumps[index]
            if offset < 0 or length < 0 or offset + length > len(data) or length % layout.size:
                raise ValueError(f"invalid AAS lump {index} in {path}")
            return [layout.unpack_from(data, offset + row * layout.size)
                    for row in range(length // layout.size)]

        raw_areas = rows(7, AREA)
        raw_settings = rows(8, SETTINGS)
        raw_reaches = rows(9, REACH)
        if len(raw_areas) != len(raw_settings):
            raise ValueError(f"AAS area/settings mismatch in {path}")
        self.areas = [Area(row[3:6], row[6:9], row[9:12]) for row in raw_areas]
        self.adjacency: list[list[Edge]] = [[] for _ in self.areas]
        for source, settings in enumerate(raw_settings):
            count, first = settings[5], settings[6]
            if count < 0 or first < 0 or first + count > len(raw_reaches):
                raise ValueError(f"invalid AAS reach span in area {source}")
            for index in range(first, first + count):
                row = raw_reaches[index]
                travel_type = row[9] & 0x00FFFFFF
                destination = row[0]
                if travel_type not in WALKABLE or destination <= 0 or destination >= len(self.areas):
                    continue
                self.adjacency[source].append(Edge(
                    destination, row[3:6], row[6:9], travel_type, row[10]))

    def area_for(self, point: tuple[float, float, float]) -> int | None:
        matches = []
        nearest = (float("inf"), None)
        for index, area in enumerate(self.areas[1:], 1):
            outside = sum(max(area.mins[axis] - point[axis], 0,
                              point[axis] - area.maxs[axis]) ** 2
                          for axis in range(3))
            if outside <= 8.0 ** 2:
                matches.append((math.dist(point, area.center), index))
            elif outside < nearest[0]:
                nearest = outside, index
        if matches:
            return min(matches)[1]
        return nearest[1] if nearest[0] <= 128.0 ** 2 else None

    def route(self, start: tuple[float, float, float],
              goal: tuple[float, float, float]) -> list[tuple[tuple[float, float, float], int]]:
        source, destination = self.area_for(start), self.area_for(goal)
        if source is None or destination is None or source == destination:
            return []
        queue = [(0, source)]
        cost = {source: 0}
        previous: dict[int, tuple[int, Edge]] = {}
        while queue:
            current_cost, area = heapq.heappop(queue)
            if current_cost != cost[area]:
                continue
            if area == destination:
                break
            for edge in self.adjacency[area]:
                penalty = 100 if edge.travel_type not in (2, 3, 7) else 0
                new_cost = current_cost + max(edge.travel_time, 1) + penalty
                if new_cost < cost.get(edge.destination, float("inf")):
                    cost[edge.destination] = new_cost
                    previous[edge.destination] = area, edge
                    heapq.heappush(queue, (new_cost, edge.destination))
        if destination not in previous:
            return []
        edges = []
        area = destination
        while area != source:
            parent, edge = previous[area]
            edges.append(edge)
            area = parent
        edges.reverse()
        waypoints = []
        for edge in edges:
            waypoints.append((edge.start, edge.travel_type))
            waypoints.append((edge.end, edge.travel_type))
        return waypoints
