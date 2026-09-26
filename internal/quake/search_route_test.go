package quake

import "testing"

func routeTravel(route []Waypoint, start, goal Vec3) float64 {
	d := 0.
	for _, w := range route {
		d += Horizontal(start, w.Position)
		start = w.Position
	}
	return d + Horizontal(start, goal)
}

func TestSearchRouteUsesDistanceAndRetainsDistinctArrivals(t *testing.T) {
	start, goal := Vec3{0, 0, 0}, Vec3{200, 80, 0}
	n := &Navigator{Areas: []Area{{},
		{Min: Vec3{-5, -5, -5}, Max: Vec3{5, 5, 5}},
		{Min: Vec3{90, -5, -5}, Max: Vec3{110, 85, 5}},
		{Min: Vec3{195, 75, -5}, Max: Vec3{205, 85, 5}}}, Edges: make([][]Edge, 4)}
	// The cheaper arrival at area 2 is farther from its exit. Collapsing all
	// arrivals to one area loses the actual shortest route via (100,80).
	n.Edges[1] = []Edge{
		{To: 2, Kind: 2, Cost: 1, Start: start, End: Vec3{100, 0, 0}},
		{To: 2, Kind: 2, Cost: 999, Start: start, End: Vec3{100, 80, 0}},
	}
	n.Edges[2] = []Edge{{To: 3, Kind: 2, Cost: 1, Start: Vec3{100, 80, 0}, End: goal}}
	old, ok := n.Route(start, goal)
	if !ok {
		t.Fatal("missing old route")
	}
	got, ok := n.SearchRoute(start, goal)
	if !ok || len(got) != 4 || got[1].Position != (Vec3{100, 80, 0}) || routeTravel(got, start, goal) >= routeTravel(old, start, goal) {
		t.Fatalf("did not select shorter physical route: %v", got)
	}
}

func TestSearchRouteCountsFinalLegAndRejectsForbiddenTravel(t *testing.T) {
	start, goal := Vec3{0, 0, 0}, Vec3{200, 0, 0}
	n := &Navigator{Areas: []Area{{},
		{Min: Vec3{-5, -5, -5}, Max: Vec3{5, 5, 5}},
		{Min: Vec3{95, -5, -5}, Max: Vec3{205, 105, 5}}}, Edges: make([][]Edge, 3)}
	// The first reached goal-area state costs less, but its final leg is longer.
	n.Edges[1] = []Edge{
		{To: 2, Kind: 2, Start: start, End: Vec3{100, 100, 0}},
		{To: 2, Kind: 2, Start: start, End: Vec3{190, 0, 0}},
		{To: 2, Kind: 11, Start: start, End: goal, Model: 1, Rise: 100},
	}
	got, ok := n.SearchRoute(start, goal)
	if !ok || len(got) != 2 || got[1].Position != (Vec3{190, 0, 0}) || routeTravel(got, start, goal) != 200 {
		t.Fatalf("wrong final leg: %v", got)
	}
	for _, kind := range []int{4, 5, 9, 11} {
		n.Edges[1] = []Edge{{To: 2, Kind: kind, Start: start, End: goal}}
		if _, ok := n.SearchRoute(start, goal); ok {
			t.Fatalf("accepted forbidden travel %d", kind)
		}
	}
}
