package harness

import (
	"math"
	"q2coopbot/internal/quake"
	"sort"
	"strconv"
)

type ObservedEnemy struct {
	ID     int        `json:"id"`
	Class  string     `json:"class"`
	Origin quake.Vec3 `json:"origin"`
}
type NearbyObservation struct {
	Kind        string  `json:"kind"`
	Entity      int     `json:"entity,omitempty"`
	Frames      int     `json:"observed_intervals"`
	MinDistance float64 `json:"min_horizontal_distance"`
}
type Stall struct {
	Step      string              `json:"step_id"`
	Start     int                 `json:"first_command_frame"`
	Detected  int                 `json:"detected_frame"`
	End       int                 `json:"last_observation_frame"`
	Intervals int                 `json:"intervals"`
	Nearby    []NearbyObservation `json:"nearby_observations"`
}
type MovementDiagnostics struct {
	ControllerHolds map[string]int `json:"controller_hold_frames"`
	Stalls          []Stall        `json:"stalls"`
}

func matchesStallEvidence(f *ExpectedFailure, d *MovementDiagnostics) bool {
	if f.MinStallFrames == 0 {
		return true
	}
	if d == nil {
		return false
	}
	for _, s := range d.Stalls {
		if s.Step != f.StepID || s.Intervals < f.MinStallFrames {
			continue
		}
		if f.NearbyKind == "" {
			return true
		}
		for _, n := range s.Nearby {
			if n.Kind == f.NearbyKind && n.Frames >= f.MinStallFrames {
				return true
			}
		}
	}
	return false
}

// This reports correlation only. Snapshots do not identify the server's
// collision entity; even a nearby teammate is not a proven physical blocker.
func diagnoseMovement(s Scenario, rows []Trace) *MovementDiagnostics {
	d := &MovementDiagnostics{ControllerHolds: map[string]int{}, Stalls: []Stall{}}
	walks := map[string]bool{}
	for _, step := range s.Steps {
		walks[step.ID] = step.Action == "walk"
	}
	var current *Stall
	nearby := map[string]NearbyObservation{}
	flush := func() {
		if current != nil && current.Intervals >= 3 {
			keys := []string{}
			for k := range nearby {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				current.Nearby = append(current.Nearby, nearby[k])
			}
			d.Stalls = append(d.Stalls, *current)
		}
		current = nil
		nearby = map[string]NearbyObservation{}
	}
	for i, row := range rows {
		if row.Scenario != nil && row.Scenario.State == "running" && row.Scenario.MovementReason != "" {
			d.ControllerHolds[row.Scenario.MovementReason]++
		}
		if i == 0 {
			continue
		}
		prev := rows[i-1]
		moving := prev.Command.Forward != 0 || prev.Command.Side != 0
		eligible := prev.Scenario != nil && walks[prev.Scenario.StepID] && prev.Scenario.State == "running" && moving && prev.OnGround != nil && *prev.OnGround && row.OnGround != nil && *row.OnGround && prev.Self != nil && row.Self != nil && row.Frame == prev.Frame+1 && row.Map == prev.Map && row.Generation == prev.Generation
		if !eligible {
			flush()
			continue
		}
		delta := quake.Distance(*prev.Self, *row.Self)
		if math.IsNaN(delta) || math.IsInf(delta, 0) || delta >= 1 {
			flush()
			continue
		}
		if current != nil && current.Step != prev.Scenario.StepID {
			flush()
		}
		if current == nil {
			current = &Stall{Step: prev.Scenario.StepID, Start: prev.Frame, Nearby: []NearbyObservation{}}
		}
		current.Intervals++
		current.End = row.Frame
		if current.Intervals == 3 {
			current.Detected = row.Frame
		}
		add := func(key, kind string, id int, p quake.Vec3) {
			distance := quake.Horizontal(*row.Self, p)
			if math.IsNaN(distance) || math.IsInf(distance, 0) || math.IsNaN(p[2]) || math.IsInf(p[2], 0) || distance > 64 || math.Abs((*row.Self)[2]-p[2]) > 48 {
				return
			}
			v, ok := nearby[key]
			if !ok {
				v = NearbyObservation{Kind: kind, Entity: id, MinDistance: distance}
			}
			v.Frames++
			v.MinDistance = math.Min(v.MinDistance, distance)
			nearby[key] = v
		}
		if row.Teammate != nil {
			add("teammate", "teammate", 0, *row.Teammate)
		}
		seen := map[int]bool{}
		for _, e := range row.Enemies {
			if seen[e.ID] {
				continue
			}
			seen[e.ID] = true
			add(e.Class+":"+strconv.Itoa(e.ID), "enemy", e.ID, e.Origin)
		}
	}
	flush()
	return d
}
