package policy

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
)

const MemoryRouteObservationVersion = "combat_observed_memory_static_routes_v1"
const MemoryRouteFeatureVersion = "combat_features_v13_memory_static_routes"
const MemoryRouteTailWidth = 65
const MemoryRouteFeatureWidth = ProjectileTypeFeatureWidth + MemoryRouteTailWidth

// Static graph evidence only. No current hidden enemy state or traversal command.
type MemoryRouteHint struct {
	ID       int         `json:"observed_id"`
	Age      int         `json:"memory_age_frames"`
	Known    bool        `json:"static_geometry_known"`
	Found    bool        `json:"static_route_found"`
	Direct   bool        `json:"static_direct_hull_clear"`
	Distance float64     `json:"static_horizontal_path_length"`
	Next     *quake.Vec3 `json:"next_static_waypoint_relative,omitempty"`
	Kind     int         `json:"static_travel_kind,omitempty"`
}
type MemoryRouteObservation struct {
	Version string            `json:"version"`
	Frame   int               `json:"frame"`
	Hints   []MemoryRouteHint `json:"hints"`
}

func memoryRouteFeatures(o Observation) ([]float64, error) {
	x := make([]float64, MemoryRouteTailWidth)
	r := o.MemoryRoutes
	if r == nil {
		return x, nil
	}
	if r.Version != MemoryRouteObservationVersion || r.Frame != o.Identity.Frame || len(r.Hints) > 8 || len(o.RememberedThreats) > 8 {
		return nil, fmt.Errorf("invalid memory route contract")
	}
	x[0] = 1
	seen := map[int]bool{}
	yaw := degrees(o.ViewAngles[1]) * math.Pi / 180
	cy, sy := math.Cos(yaw), math.Sin(yaw)
	bit := func(v bool) float64 {
		if v {
			return 1
		}
		return 0
	}
	for _, h := range r.Hints {
		if h.ID <= 0 || seen[h.ID] || h.Age < 0 || h.Age > ThreatMemoryFrames || math.IsNaN(h.Distance) || math.IsInf(h.Distance, 0) || h.Distance < 0 || h.Distance > 65536 || h.Kind < 0 || h.Kind > 16 {
			return nil, fmt.Errorf("invalid memory route hint")
		}
		seen[h.ID] = true
		slot := -1
		for i, t := range o.RememberedThreats {
			if t.ID == h.ID && t.AgeFrames == h.Age {
				slot = i
				break
			}
		}
		if slot < 0 {
			return nil, fmt.Errorf("route has no observed remembered endpoint")
		}
		if !h.Known && (h.Found || h.Direct || h.Distance != 0 || h.Next != nil || h.Kind != 0) {
			return nil, fmt.Errorf("unknown static route has values")
		}
		if h.Found != (h.Next != nil) || !h.Found && (h.Distance != 0 || h.Kind != 0) {
			return nil, fmt.Errorf("invalid static route availability")
		}
		at := 1 + slot*8
		x[at], x[at+1], x[at+2], x[at+3] = bit(h.Known), bit(h.Found), bit(h.Direct), math.Min(16, h.Distance/4096)
		if h.Next != nil {
			for _, v := range *h.Next {
				if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 32768 {
					return nil, fmt.Errorf("invalid route waypoint")
				}
			}
			x[at+4], x[at+5], x[at+6] = (h.Next[0]*cy+h.Next[1]*sy)/512, (-h.Next[0]*sy+h.Next[1]*cy)/512, h.Next[2]/512
		}
		x[at+7] = float64(h.Kind) / 16
	}
	return x, nil
}
