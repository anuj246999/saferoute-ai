"""
Module 5 — Routing Logic using Dijkstra's Algorithm.

Implements shortest-path routing where edge cost considers both
distance and safety risk score.

Edge cost formula: cost = distance + risk_score
The algorithm prefers routes with lower total safety risk.
"""

import heapq
from dataclasses import dataclass, field


@dataclass
class Edge:
    """Represents a road segment between two nodes."""

    to_node: str
    distance: float
    risk_score: float

    @property
    def cost(self) -> float:
        """Combined cost = distance + risk_score."""
        return self.distance + self.risk_score


@dataclass
class RouteResult:
    """Result of a route computation."""

    path: list[str]
    total_distance: float
    total_risk: float
    total_cost: float
    segment_details: list[dict]

    def __str__(self) -> str:
        path_str = " → ".join(self.path)
        return (
            f"Route: {path_str}\n"
            f"  Total Distance: {self.total_distance:.1f}\n"
            f"  Total Risk: {self.total_risk:.1f}\n"
            f"  Total Cost: {self.total_cost:.1f}"
        )


class SafeRouteGraph:
    """
    Graph representation for road network with safety-aware routing.

    Uses Dijkstra's algorithm with cost = distance + risk_score
    to find the safest route between locations.
    """

    def __init__(self):
        """Initialize an empty road network graph."""
        self.adjacency: dict[str, list[Edge]] = {}

    def add_node(self, node: str) -> None:
        """Add a node (intersection) to the graph."""
        if node not in self.adjacency:
            self.adjacency[node] = []

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        distance: float,
        risk_score: float,
        bidirectional: bool = True,
    ) -> None:
        """
        Add a road segment (edge) between two nodes.

        Args:
            from_node: Starting intersection.
            to_node: Ending intersection.
            distance: Physical distance of the road segment.
            risk_score: Safety risk score (0-10).
            bidirectional: If True, adds edge in both directions.
        """
        self.add_node(from_node)
        self.add_node(to_node)

        self.adjacency[from_node].append(Edge(to_node, distance, risk_score))
        if bidirectional:
            self.adjacency[to_node].append(Edge(from_node, distance, risk_score))

    def update_risk_scores(self, risk_scores: dict[tuple[str, str], float]) -> None:
        """
        Update risk scores for edges based on new safety analysis.

        Args:
            risk_scores: Mapping of (from_node, to_node) to new risk scores.
        """
        for (from_node, to_node), risk in risk_scores.items():
            if from_node in self.adjacency:
                for edge in self.adjacency[from_node]:
                    if edge.to_node == to_node:
                        edge.risk_score = risk

    def find_safest_route(self, start: str, end: str) -> RouteResult | None:
        """
        Find the safest route using Dijkstra's algorithm.

        Cost = distance + risk_score for each edge.

        Args:
            start: Starting node.
            end: Destination node.

        Returns:
            RouteResult with the safest path, or None if no path exists.
        """
        if start not in self.adjacency or end not in self.adjacency:
            return None

        # dist[node] = minimum cost to reach node
        dist: dict[str, float] = {node: float("inf") for node in self.adjacency}
        dist[start] = 0.0

        # Track predecessors for path reconstruction
        prev: dict[str, str | None] = {node: None for node in self.adjacency}
        prev_edge: dict[str, Edge | None] = {node: None for node in self.adjacency}

        # Priority queue: (cost, node)
        pq: list[tuple[float, str]] = [(0.0, start)]
        visited: set[str] = set()

        while pq:
            current_cost, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == end:
                break

            for edge in self.adjacency[current]:
                if edge.to_node in visited:
                    continue

                new_cost = current_cost + edge.cost
                if new_cost < dist[edge.to_node]:
                    dist[edge.to_node] = new_cost
                    prev[edge.to_node] = current
                    prev_edge[edge.to_node] = edge
                    heapq.heappush(pq, (new_cost, edge.to_node))

        # Reconstruct path
        if dist[end] == float("inf"):
            return None

        path = []
        segment_details = []
        current = end

        while current is not None:
            path.append(current)
            edge = prev_edge[current]
            if edge is not None:
                segment_details.append({
                    "from": prev[current],
                    "to": current,
                    "distance": edge.distance,
                    "risk_score": edge.risk_score,
                    "cost": edge.cost,
                })
            current = prev[current]

        path.reverse()
        segment_details.reverse()

        total_distance = sum(s["distance"] for s in segment_details)
        total_risk = sum(s["risk_score"] for s in segment_details)
        total_cost = sum(s["cost"] for s in segment_details)

        return RouteResult(
            path=path,
            total_distance=round(total_distance, 1),
            total_risk=round(total_risk, 1),
            total_cost=round(total_cost, 1),
            segment_details=segment_details,
        )


def create_sample_graph(risk_scores: dict[str, float] | None = None) -> SafeRouteGraph:
    """
    Create a sample road network graph for demonstration.

    The graph represents a small urban area with multiple route options.

    Args:
        risk_scores: Optional mapping of road_segment names to risk scores.
            If provided, these override the default risk scores.

    Returns:
        A SafeRouteGraph with sample nodes and edges.
    """
    graph = SafeRouteGraph()

    # Default risk scores if not provided
    default_risks = {
        "A-B": 3.0,
        "A-C": 2.0,
        "B-D": 7.0,
        "B-E": 4.0,
        "C-D": 5.0,
        "C-F": 1.5,
        "D-G": 6.0,
        "E-G": 2.5,
        "F-E": 3.0,
        "F-G": 4.0,
    }

    risks = default_risks.copy()
    if risk_scores:
        risks.update(risk_scores)

    # Build the graph
    # Edges: (from, to, distance, risk_score)
    edges = [
        ("A", "B", 2.0, risks["A-B"]),
        ("A", "C", 3.0, risks["A-C"]),
        ("B", "D", 4.0, risks["B-D"]),
        ("B", "E", 3.0, risks["B-E"]),
        ("C", "D", 2.0, risks["C-D"]),
        ("C", "F", 4.0, risks["C-F"]),
        ("D", "G", 3.0, risks["D-G"]),
        ("E", "G", 2.0, risks["E-G"]),
        ("F", "E", 2.0, risks["F-E"]),
        ("F", "G", 3.0, risks["F-G"]),
    ]

    for from_node, to_node, distance, risk in edges:
        graph.add_edge(from_node, to_node, distance, risk)

    return graph


if __name__ == "__main__":
    print("=== SafeRoute AI - Routing Demo ===\n")

    graph = create_sample_graph()
    result = graph.find_safest_route("A", "G")

    if result:
        print(result)
        print("\nSegment Details:")
        for seg in result.segment_details:
            print(f"  {seg['from']} → {seg['to']}: "
                  f"distance={seg['distance']}, risk={seg['risk_score']}, cost={seg['cost']}")
    else:
        print("No route found!")
