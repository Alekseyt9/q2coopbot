package policy

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
)

const NavigationFeatureVersion = "combat_features_v8"
const NavigationFeatureWidth = 881

// Frozen order: presence, eight goals, four statuses, three masked XYZ
// positions, masked age. Positions use current yaw, units/512, bounded to 4.
func navigationFeatures(o Observation) ([]float64, error) {
	v := make([]float64, 27)
	n := o.Navigation
	if n == nil {
		return v, nil
	}
	v[0] = 1
	for i, name := range []string{"follow_teammate", "cover_teammate", "search_last_seen", "probe_last_seen", "wait_for_teammate", "collect_item", "recover_health", "reach_level_exit"} {
		if n.Goal == name {
			v[1+i] = 1
		}
	}
	for i, name := range []string{"ready", "direct_clear", "unreachable", "aas_missing"} {
		if n.Status == name {
			v[9+i] = 1
		}
	}
	yaw := degrees(o.ViewAngles[1]) * math.Pi / 180
	for i, p := range []*quake.Vec3{n.GoalRelative, n.WaypointRelative, n.LastTeammateRelative} {
		if p == nil {
			continue
		}
		for _, x := range *p {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				return nil, fmt.Errorf("nonfinite navigation position")
			}
		}
		at := 13 + i*4
		v[at] = 1
		local := [3]float64{p[0]*math.Cos(yaw) + p[1]*math.Sin(yaw), -p[0]*math.Sin(yaw) + p[1]*math.Cos(yaw), p[2]}
		for j, x := range local {
			v[at+1+j] = math.Max(-4, math.Min(4, x/512))
		}
	}
	if n.LastTeammateAgeFrames != nil {
		if *n.LastTeammateAgeFrames < 0 {
			return nil, fmt.Errorf("negative navigation age")
		}
		v[25] = 1
		v[26] = math.Min(1, float64(*n.LastTeammateAgeFrames)/200)
	}
	return v, nil
}

// Optional observed navigation context. Feature versions v1-v7 deliberately
// ignore it; v8 appends masked navigation features.
// A remembered position is explicitly distinct from a currently seen player.
type NavigationContext struct {
	Goal                  string      `json:"goal"`
	Status                string      `json:"status"`
	GoalRelative          *quake.Vec3 `json:"goal_relative,omitempty"`
	WaypointRelative      *quake.Vec3 `json:"waypoint_relative,omitempty"`
	LastTeammateRelative  *quake.Vec3 `json:"last_teammate_relative,omitempty"`
	LastTeammateAgeFrames *int        `json:"last_teammate_age_frames,omitempty"`
}
