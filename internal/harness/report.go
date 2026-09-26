package harness

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"

	"q2coopbot/internal/quake"
)

type Trace struct {
	Map        string        `json:"map"`
	Generation int           `json:"spawncount"`
	Frame      int           `json:"frame"`
	Teammate   *quake.Vec3   `json:"teammate"`
	Goal       string        `json:"goal"`
	Command    quake.UserCmd `json:"sent_command"`
	Scenario   *Status       `json:"scenario"`
}
type Event struct {
	Frame int    `json:"frame"`
	Kind  string `json:"kind"`
}
type Report struct {
	Metrics Metrics `json:"metrics"`
	State             string  `json:"state"`
	Reason            string  `json:"reason,omitempty"`
	Frame             int     `json:"first_problem_frame,omitempty"`
	Events            []Event `json:"events"`
	Losses            int     `json:"contact_losses"`
	Reacquisitions    int     `json:"reacquisitions"`
	FollowResumptions int     `json:"follow_resumptions"`
	CompletedSteps    int     `json:"completed_steps"`
}

// Durations use server frames (10 Hz), independent of wall-clock acceleration.
type Metrics struct {
	Frames int `json:"analyzed_frames"`
	VisibleFrames int `json:"visible_frames"`
	MovingCommandFrames int `json:"moving_command_frames"`
	GoalFrames map[string]int `json:"goal_frames"`
	RecoveryFrames []int `json:"contact_recovery_frames"`
	FollowDelayFrames []int `json:"follow_delay_frames"`
	GameSeconds float64 `json:"game_seconds"`
}

func ReadTrace(path string) ([]Trace, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []Trace
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 4096), 4*1024*1024)
	for s.Scan() {
		var row Trace
		if err := json.Unmarshal(s.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("trace line %d: %w", len(rows)+1, err)
		}
		rows = append(rows, row)
	}
	return rows, s.Err()
}
func Analyze(s Scenario, actor, bot []Trace) Report {
	r := Report{State: "fixture_failed", Reason: "scenario_incomplete"}
	end := 0
	for _, row := range actor {
		if row.Scenario == nil {
			continue
		}
		if row.Scenario.State == "failed" {
			r.Frame = row.Frame
			r.Reason = row.Scenario.Reason
			return r
		}
		if row.Scenario.State == "completed" {
			end = row.Scenario.EndFrame
			r.CompletedSteps = row.Scenario.CompletedSteps
			break
		}
	}
	if end < s.StartFrame || r.CompletedSteps != len(s.Steps) {
		return r
	}
	filter := func(rows []Trace) ([]Trace, error) {
		var selected []Trace
		for _, row := range rows {
			if row.Map == s.Map && row.Frame >= s.StartFrame && row.Frame <= end {
				selected = append(selected, row)
			}
		}
		if len(selected) != end-s.StartFrame+1 {
			return nil, fmt.Errorf("incomplete frame window")
		}
		for i, row := range selected {
			if row.Frame != s.StartFrame+i || row.Generation != selected[0].Generation {
				return nil, fmt.Errorf("frame/generation discontinuity at %d", row.Frame)
			}
		}
		return selected, nil
	}
	actorRows, err := filter(actor)
	if err != nil {
		r.State = "trace_invalid"
		r.Reason = "actor: " + err.Error()
		return r
	}
	rows, err := filter(bot)
	if err != nil {
		r.State = "trace_invalid"
		r.Reason = "bot: " + err.Error()
		return r
	}
	if actorRows[0].Generation != rows[0].Generation {
		r.State, r.Reason = "trace_invalid", "actor/bot generation mismatch"
		return r
	}
	r.Metrics = Metrics{Frames: len(rows), GameSeconds: float64(len(rows))/10, GoalFrames: map[string]int{}, RecoveryFrames: []int{}, FollowDelayFrames: []int{}}
	lossFrame, recoveryFrame := 0, 0
	seen, lost, awaitFollow := false, false, false
	for _, row := range rows {
		visible := row.Teammate != nil
		r.Metrics.GoalFrames[row.Goal]++
		if visible { r.Metrics.VisibleFrames++ }
		if row.Command.Forward != 0 || row.Command.Side != 0 { r.Metrics.MovingCommandFrames++ }
		if seen && !visible && !lost {
			lost = true
			awaitFollow = false
			r.Losses++
			lossFrame = row.Frame
			r.Events = append(r.Events, Event{row.Frame, "contact_lost"})
		}
		if visible && lost {
			lost = false
			awaitFollow = true
			r.Reacquisitions++
			recoveryFrame = row.Frame
			r.Metrics.RecoveryFrames = append(r.Metrics.RecoveryFrames, row.Frame-lossFrame)
			r.Events = append(r.Events, Event{row.Frame, "contact_reacquired"})
		}
		if visible {
			seen = true
		}
		if awaitFollow && visible && row.Goal == "follow_teammate" && (row.Command.Forward != 0 || row.Command.Side != 0) {
			awaitFollow = false
			r.FollowResumptions++
			r.Metrics.FollowDelayFrames = append(r.Metrics.FollowDelayFrames, row.Frame-recoveryFrame)
			r.Events = append(r.Events, Event{row.Frame, "following_resumed"})
		}
	}
	r.State = "passed"
	r.Reason = ""
	if r.Losses < s.Expect.ContactLosses || r.Reacquisitions < s.Expect.Reacquisitions || r.FollowResumptions < s.Expect.FollowResumptions {
		r.State = "behavior_failed"
		r.Reason = "contact cycle expectations not met"
		r.Frame = end
	}
	return r
}
