package quake

import (
	"container/heap"
	"math"
)

type distanceItem struct {
	state    int
	distance float64
}
type distanceQueue []distanceItem

func (q distanceQueue) Len() int { return len(q) }
func (q distanceQueue) Less(i, j int) bool {
	if q[i].distance == q[j].distance {
		return q[i].state < q[j].state
	}
	return q[i].distance < q[j].distance
}
func (q distanceQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *distanceQueue) Push(x any)   { *q = append(*q, x.(distanceItem)) }
func (q *distanceQueue) Pop() any     { a := *q; x := a[len(a)-1]; *q = a[:len(a)-1]; return x }

// SearchRoute minimizes horizontal waypoint travel, including the start and
// final legs. Each arrival edge is a distinct state: two entries into one area
// can have different costs to its next exit. AAS travel-time costs are not used.
// The search policy excludes jumps and elevators, as does safeSearchRoute.
func (n *Navigator) SearchRoute(start, goal Vec3) ([]Waypoint, bool) {
	src, dst := n.AreaFor(start), n.AreaFor(goal)
	if src < 0 || dst < 0 {
		return nil, false
	}
	if src == dst {
		return []Waypoint{}, true
	}
	edges := []Edge{{To: src, End: start}} // virtual arrival at the actual start
	out := make([][]int, len(n.Areas))
	for from, list := range n.Edges {
		for _, e := range list {
			if e.Kind != 2 && e.Kind != 3 && e.Kind != 6 && e.Kind != 7 && e.Kind != 8 {
				continue
			}
			out[from] = append(out[from], len(edges))
			edges = append(edges, e)
		}
	}
	dist := make([]float64, len(edges))
	prev := make([]int, len(edges))
	for i := range dist {
		dist[i] = math.Inf(1)
	}
	dist[0] = 0
	q := &distanceQueue{{state: 0, distance: 0}}
	heap.Init(q)
	best, finish := math.Inf(1), -1
	for q.Len() > 0 {
		cur := heap.Pop(q).(distanceItem)
		if cur.distance != dist[cur.state] {
			continue
		}
		if cur.distance >= best {
			break
		}
		at := edges[cur.state]
		if at.To == dst {
			total := cur.distance + Horizontal(at.End, goal)
			if total < best {
				best, finish = total, cur.state
			}
			continue
		}
		for _, id := range out[at.To] {
			e := edges[id]
			next := cur.distance + Horizontal(at.End, e.Start) + Horizontal(e.Start, e.End)
			if next < dist[id] {
				dist[id] = next
				prev[id] = cur.state
				heap.Push(q, distanceItem{state: id, distance: next})
			}
		}
	}
	if finish < 0 {
		return nil, false
	}
	var reverse []Edge
	for at := finish; at != 0; at = prev[at] {
		reverse = append(reverse, edges[at])
	}
	waypoints := make([]Waypoint, 0, len(reverse)*2)
	for i := len(reverse) - 1; i >= 0; i-- {
		e := reverse[i]
		waypoints = append(waypoints, Waypoint{Position: e.Start, Kind: e.Kind, ToArea: e.To}, Waypoint{Position: e.End, Kind: e.Kind, ToArea: e.To})
	}
	return waypoints, true
}
