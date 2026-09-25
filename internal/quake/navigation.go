package quake

import (
	"container/heap"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
)

type Vec3 [3]float64
type Area struct {
	Min, Max, Center Vec3
	Contents, Flags  int
}
type Edge struct {
	To          int
	Start, End  Vec3
	Kind, Cost  int
	Model, Rise int
}
type Waypoint struct {
	Position      Vec3   `json:"position"`
	Jump          bool   `json:"jump"`
	Kind          int    `json:"kind,omitempty"`
	Model         int    `json:"model,omitempty"`
	Rise          int    `json:"rise,omitempty"`
	ToArea        int    `json:"to_area,omitempty"`
	ElevatorPhase string `json:"elevator_phase,omitempty"`
}
type Navigator struct {
	Areas []Area
	Edges [][]Edge
}

func vec(data []byte, at int) Vec3 {
	var v Vec3
	for i := range v {
		v[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(data[at+i*4:])))
	}
	return v
}
func LoadAAS(path string) (*Navigator, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 120 || string(data[:4]) != "EAAS" {
		return nil, errors.New("invalid AAS header")
	}
	version := binary.LittleEndian.Uint32(data[4:])
	if version != 2 && version != 3 && version != 4 && version != 5 {
		return nil, fmt.Errorf("unsupported AAS version %d", version)
	}
	headerSize := 8 + 14*8
	if version >= 4 {
		headerSize += 4 // BSP checksum precedes the lump directory.
	}
	if len(data) < headerSize {
		return nil, errors.New("truncated AAS header")
	}
	header := make([]byte, headerSize)
	copy(header, data[:headerSize])
	if version == 5 {
		for i := 8; i < len(header); i++ {
			header[i] ^= byte((i - 8) * 119)
		}
	}
	lump := func(i, rowSize int) ([]byte, error) {
		p := headerSize - 14*8 + i*8
		off := int(int32(binary.LittleEndian.Uint32(header[p:])))
		n := int(int32(binary.LittleEndian.Uint32(header[p+4:])))
		if off < headerSize || n < 0 || off > len(data) || n > len(data)-off || n%rowSize != 0 {
			return nil, fmt.Errorf("invalid AAS lump %d", i)
		}
		return data[off : off+n], nil
	}
	a, e := lump(7, 48)
	if e != nil {
		return nil, e
	}
	s, e := lump(8, 28)
	if e != nil {
		return nil, e
	}
	r, e := lump(9, 44)
	if e != nil {
		return nil, e
	}
	if len(a)/48 != len(s)/28 {
		return nil, errors.New("AAS area/settings mismatch")
	}
	n := &Navigator{Areas: make([]Area, len(a)/48), Edges: make([][]Edge, len(a)/48)}
	for i := range n.Areas {
		p := i * 48
		n.Areas[i] = Area{Min: vec(a, p+12), Max: vec(a, p+24), Center: vec(a, p+36),
			Contents: int(int32(binary.LittleEndian.Uint32(s[i*28:]))),
			Flags:    int(int32(binary.LittleEndian.Uint32(s[i*28+4:])))}
	}
	for i := range n.Edges {
		p := i * 28
		count := int(int32(binary.LittleEndian.Uint32(s[p+20:])))
		first := int(int32(binary.LittleEndian.Uint32(s[p+24:])))
		if first < 0 || count < 0 || first > len(r)/44 || count > len(r)/44-first {
			return nil, fmt.Errorf("invalid reach span at area %d", i)
		}
		for j := first; j < first+count; j++ {
			p = j * 44
			to := int(int32(binary.LittleEndian.Uint32(r[p:])))
			kind := int(binary.LittleEndian.Uint32(r[p+36:]) & 0xffffff)
			cost := int(binary.LittleEndian.Uint16(r[p+40:]))
			if to > 0 && to < len(n.Areas) && (kind >= 2 && kind <= 9 || kind == 11) {
				model, rise := 0, 0
				if kind == 11 {
					model = int(int32(binary.LittleEndian.Uint32(r[p+4:])))
					rise = int(int32(binary.LittleEndian.Uint32(r[p+8:])))
					if model <= 0 || rise <= 0 {
						continue
					}
				}
				n.Edges[i] = append(n.Edges[i], Edge{To: to, Start: vec(r, p+12), End: vec(r, p+24), Kind: kind, Cost: cost, Model: model, Rise: rise})
			}
		}
	}
	return n, nil
}
func Distance(a, b Vec3) float64 {
	x, y, z := a[0]-b[0], a[1]-b[1], a[2]-b[2]
	return math.Sqrt(x*x + y*y + z*z)
}
func Horizontal(a, b Vec3) float64 { return math.Hypot(a[0]-b[0], a[1]-b[1]) }
func (n *Navigator) AreaFor(p Vec3) int {
	best, score := -1, math.Inf(1)
	nearby := -1
	nearScore := math.Inf(1)
	for i := 1; i < len(n.Areas); i++ {
		a := n.Areas[i]
		outside := 0.0
		for j := 0; j < 3; j++ {
			d := math.Max(a.Min[j]-p[j], math.Max(0, p[j]-a.Max[j]))
			outside += d * d
		}
		if outside <= 64 {
			d := Distance(p, a.Center)
			if d < score {
				best, score = i, d
			}
		} else if outside < nearScore {
			nearby, nearScore = i, outside
		}
	}
	if best >= 0 {
		return best
	}
	if nearScore <= 128*128 {
		return nearby
	}
	return -1
}

// GroundedNear only trusts areas that actually contain the point (allowing
// small BSP/AAS rounding differences), unlike AreaFor's route fallback.
func (n *Navigator) GroundedNear(p Vec3) bool {
	if n == nil {
		return false
	}
	for i := 1; i < len(n.Areas); i++ {
		a := n.Areas[i]
		if a.Flags&1 == 0 {
			continue
		}
		outside := 0.0
		for axis := 0; axis < 3; axis++ {
			d := math.Max(a.Min[axis]-p[axis], math.Max(0, p[axis]-a.Max[axis]))
			outside += d * d
		}
		if outside <= 4 {
			return true
		}
	}
	return false
}

type queueItem struct{ area, cost int }
type routeQueue []queueItem

func (q routeQueue) Len() int           { return len(q) }
func (q routeQueue) Less(i, j int) bool { return q[i].cost < q[j].cost }
func (q routeQueue) Swap(i, j int)      { q[i], q[j] = q[j], q[i] }
func (q *routeQueue) Push(x any)        { *q = append(*q, x.(queueItem)) }
func (q *routeQueue) Pop() any          { old := *q; x := old[len(old)-1]; *q = old[:len(old)-1]; return x }
func (n *Navigator) Route(start, goal Vec3) ([]Waypoint, bool) {
	src, dst := n.AreaFor(start), n.AreaFor(goal)
	if src < 0 || dst < 0 {
		return nil, false
	}
	if src == dst {
		return []Waypoint{}, true
	}
	cost := make([]int, len(n.Areas))
	for i := range cost {
		cost[i] = math.MaxInt
	}
	cost[src] = 0
	prev := make([]int, len(n.Areas))
	edges := make([]Edge, len(n.Areas))
	q := &routeQueue{{src, 0}}
	heap.Init(q)
	for q.Len() > 0 {
		cur := heap.Pop(q).(queueItem)
		if cur.cost != cost[cur.area] {
			continue
		}
		if cur.area == dst {
			break
		}
		for _, edge := range n.Edges[cur.area] {
			penalty := 0
			if edge.Kind != 2 && edge.Kind != 3 && edge.Kind != 7 {
				penalty = 100
			}
			next := cur.cost + max(edge.Cost, 1) + penalty
			if next < cost[edge.To] {
				cost[edge.To] = next
				prev[edge.To] = cur.area
				edges[edge.To] = edge
				heap.Push(q, queueItem{edge.To, next})
			}
		}
	}
	if cost[dst] == math.MaxInt {
		return nil, false
	}
	var reverse []Edge
	for at := dst; at != src; at = prev[at] {
		reverse = append(reverse, edges[at])
	}
	waypoints := make([]Waypoint, 0, len(reverse)*2)
	for i := len(reverse) - 1; i >= 0; i-- {
		e := reverse[i]
		jump := e.Kind == 4 || e.Kind == 5 || e.Kind == 9
		if e.Kind == 11 {
			waypoints = append(waypoints,
				Waypoint{Position: e.Start, Kind: 11, Model: e.Model, Rise: e.Rise, ToArea: e.To, ElevatorPhase: "board"},
				Waypoint{Position: e.End, Kind: 11, Model: e.Model, Rise: e.Rise, ToArea: e.To, ElevatorPhase: "exit"})
		} else {
			waypoints = append(waypoints,
				Waypoint{Position: e.Start, Jump: jump, Kind: e.Kind, ToArea: e.To},
				Waypoint{Position: e.End, Jump: jump, Kind: e.Kind, ToArea: e.To})
		}
	}
	return waypoints, true
}
