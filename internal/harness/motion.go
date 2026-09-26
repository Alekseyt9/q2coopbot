package harness

import "math"

type WalkMetrics struct {
	StepID          string  `json:"step_id"`
	StartFrame      int     `json:"start_frame"`
	LastFrame       int     `json:"last_frame"`
	Completed       bool    `json:"completed"`
	InitialDistance float64 `json:"initial_distance_units"`
	FinalDistance   float64 `json:"final_distance_units"`
	Progress        float64 `json:"distance_reduction_units"`
}

func measureWalks(s Scenario, rows []Trace) []WalkMetrics {
	var result []WalkMetrics
	for index, step := range s.Steps {
		if step.Action != "walk" || step.Target == nil {
			continue
		}
		var m *WalkMetrics
		valid := true
		for _, row := range rows {
			if row.Scenario == nil || row.Scenario.StepID != step.ID {
				continue
			}
			if row.Self == nil {
				valid = false
				break
			}
			d := 0.0
			for k := range *row.Self {
				delta := (*row.Self)[k] - (*step.Target)[k]
				d += delta * delta
			}
			d = math.Sqrt(d)
			if math.IsNaN(d) || math.IsInf(d, 0) {
				valid = false
				break
			}
			if m == nil {
				m = &WalkMetrics{StepID: step.ID, StartFrame: row.Frame, InitialDistance: d}
			}
			m.LastFrame = row.Frame
			m.FinalDistance = d
			m.Progress = m.InitialDistance - d
			m.Completed = row.Scenario.CompletedSteps > index
			if m.Completed {
				break
			}
		}
		if m != nil && valid {
			result = append(result, *m)
		}
	}
	return result
}

// Observed horizontal displacement, not planner intent. Low displacement is
// diagnostic only: collisions, turning and scripted holds need context.
type MotionMetrics struct {
	ExcludedDeadIntervals      int     `json:"excluded_dead_intervals"`
	ExcludedRespawnIntervals   int     `json:"excluded_respawn_intervals"`
	PathUnits                  float64 `json:"horizontal_path_units"`
	MeasuredIntervals          int     `json:"measured_intervals"`
	ExcludedPlacementIntervals int     `json:"excluded_placement_intervals"`
	CommandedIntervals         int     `json:"commanded_intervals"`
	LowDisplacementIntervals   int     `json:"low_displacement_intervals"`
	LongestLowDisplacementRun  int     `json:"longest_low_displacement_run_frames"`
}

func measureMotion(s Scenario, rows []Trace, actor bool) *MotionMetrics {
	if len(rows) < 2 {
		return nil
	}
	for _, row := range rows {
		if row.Self == nil {
			return nil
		}
		for _, v := range *row.Self {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil
			}
		}
	}
	placement := map[string]bool{}
	respawn := map[string]bool{}
	for _, step := range s.Steps {
		placement[step.ID] = step.Action == "place"
		respawn[step.ID] = step.Action == "respawn_cycle"
	}
	m := &MotionMetrics{}
	run := 0
	for i := 1; i < len(rows); i++ {
		prev, row := rows[i-1], rows[i]
		if (prev.Health != nil && *prev.Health <= 0) || (row.Health != nil && *row.Health <= 0) {
			m.ExcludedDeadIntervals++
			run = 0
			continue
		}
		if actor && prev.Scenario != nil && respawn[prev.Scenario.StepID] {
			m.ExcludedRespawnIntervals++
			run = 0
			continue
		}
		// Position at frame N is the result of command N-1. Exclude the whole
		// acknowledged placement step, including its arrival, from walking metrics.
		if actor && prev.Scenario != nil && placement[prev.Scenario.StepID] {
			m.ExcludedPlacementIntervals++
			run = 0
			continue
		}
		d := math.Hypot((*row.Self)[0]-(*prev.Self)[0], (*row.Self)[1]-(*prev.Self)[1])
		m.PathUnits += d
		m.MeasuredIntervals++
		if prev.Command.Forward != 0 || prev.Command.Side != 0 {
			m.CommandedIntervals++
			if d < 1 {
				m.LowDisplacementIntervals++
				run++
				m.LongestLowDisplacementRun = max(m.LongestLowDisplacementRun, run)
			} else {
				run = 0
			}
		} else {
			run = 0
		}
	}
	return m
}
