package harness

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"

	"q2coopbot/internal/quake"
)

type Trace struct {
	Arbitration struct { LimitReason string `json:"limit_reason"` } `json:"arbitration"`
	Connection        int             `json:"connection,omitempty"`
	ObserverKill      bool            `json:"test_observer_kill,omitempty"`
	ObserverRespawn   bool            `json:"test_observer_respawn,omitempty"`
	ClientSequence    uint32          `json:"client_sequence"`
	SessionStartFrame int             `json:"session_start_frame,omitempty"`
	Session           *SessionStatus  `json:"session,omitempty"`
	SelfEntity        int             `json:"self_entity"`
	TeammateEntity    int             `json:"teammate_entity,omitempty"`
	Health            *int16          `json:"health,omitempty"`
	OnGround          *bool           `json:"on_ground,omitempty"`
	Enemies           []ObservedEnemy `json:"enemies,omitempty"`
	Self              *quake.Vec3     `json:"self,omitempty"`
	TeammateAgeFrames *int            `json:"teammate_age_frames,omitempty"`
	SearchTarget      *quake.Vec3     `json:"search_target,omitempty"`
	SearchAttempt     *SearchAttempt  `json:"search_attempt,omitempty"`
	Map               string          `json:"map"`
	Generation        int             `json:"spawncount"`
	Frame             int             `json:"frame"`
	Teammate          *quake.Vec3     `json:"teammate"`
	Goal              string          `json:"goal"`
	Command           quake.UserCmd   `json:"sent_command"`
	Scenario          *Status         `json:"scenario"`
}
type Event struct {
	Frame int    `json:"frame"`
	Kind  string `json:"kind"`
}
type Report struct {
	ProblemLocation   *FrameLocation  `json:"problem_location,omitempty"`
	Timeline          SessionTimeline `json:"session_timeline"`
	Accepted          bool            `json:"accepted"`
	Expectation       string          `json:"expectation"`
	FailedStep        string          `json:"failed_step,omitempty"`
	Checks            []Check         `json:"checks,omitempty"`
	Context           []ContextFrame  `json:"problem_context,omitempty"`
	Metrics           Metrics         `json:"metrics"`
	State             string          `json:"state"`
	Reason            string          `json:"reason,omitempty"`
	Frame             int             `json:"first_problem_frame,omitempty"`
	Events            []Event         `json:"events"`
	Losses            int             `json:"contact_losses"`
	Reacquisitions    int             `json:"reacquisitions"`
	FollowResumptions int             `json:"follow_resumptions"`
	CompletedSteps    int             `json:"completed_steps"`
}

// Durations use server frames (10 Hz), independent of wall-clock acceleration.
type Metrics struct {
	CompanionLifecycle  []CompanionCycle     `json:"companion_lifecycle,omitempty"`
	ActorLifecycle      []Event              `json:"actor_lifecycle,omitempty"`
	ActorDiagnostics    *MovementDiagnostics `json:"actor_diagnostics"`
	WalkSteps           []WalkMetrics        `json:"walk_steps,omitempty"`
	ActorMotion         *MotionMetrics       `json:"actor_motion"`
	BotMotion           *MotionMetrics       `json:"bot_motion"`
	Frames              int                  `json:"analyzed_frames"`
	VisibleFrames       int                  `json:"visible_frames"`
	MovingCommandFrames int                  `json:"moving_command_frames"`
	GoalFrames          map[string]int       `json:"goal_frames"`
	RecoveryFrames      []int                `json:"contact_recovery_frames"`
	FollowDelayFrames   []int                `json:"follow_delay_frames"`
	GameSeconds         float64              `json:"game_seconds"`
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
	r := analyze(s, actor, bot)
	r.Timeline = SessionTimeline{Actor: traceTimeline(actor), Bot: traceTimeline(bot)}
	if len(s.Expect.MapSequence) > 0 {
		if reason, location := verifyMapSequence(s.Expect.MapSequence, r.Timeline); reason != "" {
			r.State, r.Reason, r.ProblemLocation = "trace_invalid", reason, location
		}
	}
	r.Accepted = r.State == "passed"
	r.Expectation = "normal_completion"
	if f := s.Expect.Failure; f != nil {
		r.Accepted = r.State == "fixture_failed" && r.Reason == f.Reason && r.FailedStep == f.StepID && r.Frame >= f.MinFrame && r.Frame <= f.MaxFrame && r.Metrics.Frames > 0
		r.Expectation = "expected_failure_mismatch"
		if r.Accepted {
			r.Accepted = matchesStallEvidence(f, r.Metrics.ActorDiagnostics)
		}
		if r.Accepted {
			r.Expectation = "expected_failure_matched"
		}
	}
	return r
}

func analyze(s Scenario, actor, bot []Trace) Report {
	r := Report{State: "fixture_failed", Reason: "scenario_incomplete"}
	end := 0
	fixtureFailed := false
	for index, row := range actor {
		if row.Scenario == nil {
			continue
		}
		if row.Scenario.State == "failed" {
			r.ProblemLocation = &FrameLocation{Row: index + 1, Map: row.Map, Generation: row.Generation, Frame: row.Frame, Connection: row.Connection}
			r.FailedStep = row.Scenario.StepID
			r.Frame = row.Frame
			r.Reason = row.Scenario.Reason
			end = row.Frame
			r.CompletedSteps = row.Scenario.CompletedSteps
			fixtureFailed = true
			if row.Scenario.Reason == "map_generation_changed" {
				r.State = "trace_invalid"
				return r
			}
			break
		}
		if row.Scenario.State == "completed" {
			end = row.Scenario.EndFrame
			r.CompletedSteps = row.Scenario.CompletedSteps
			break
		}
	}
	if end < s.StartFrame || end > s.GameFrames || !fixtureFailed && r.CompletedSteps != len(s.Steps) {
		return r
	}
	filter := func(rows []Trace) ([]Trace, error) {
		var selected []Trace
		lastIndex := -1
		for index, row := range rows {
			if row.Map == s.Map && row.Frame >= s.StartFrame && row.Frame <= end {
				if lastIndex >= 0 && index != lastIndex+1 {
					return nil, fmt.Errorf("scenario window interrupted at trace row %d", lastIndex+2)
				}
				selected = append(selected, row)
				lastIndex = index
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
	r.Metrics = Metrics{Frames: len(rows), GameSeconds: float64(len(rows)) / 10, GoalFrames: map[string]int{}, RecoveryFrames: []int{}, FollowDelayFrames: []int{}}
	r.Metrics.ActorMotion = measureMotion(s, actorRows, true)
	if r.Metrics.ActorMotion != nil {
		r.Metrics.ActorDiagnostics = diagnoseMovement(s, actorRows)
	}
	r.Metrics.BotMotion = measureMotion(s, rows, false)
	r.Metrics.WalkSteps = measureWalks(s, actorRows)
	r.Metrics.ActorLifecycle = actorLifecycle(actorRows)
	r.Metrics.CompanionLifecycle = companionLifecycle(actorRows, rows)
	lossFrame, recoveryFrame := 0, 0
	seen, lost, awaitFollow := false, false, false
	for _, row := range rows {
		visible := row.Teammate != nil
		r.Metrics.GoalFrames[row.Goal]++
		if visible {
			r.Metrics.VisibleFrames++
		}
		if row.Command.Forward != 0 || row.Command.Side != 0 {
			r.Metrics.MovingCommandFrames++
		}
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
	if fixtureFailed {
		if !checkLifecycle(s, actorRows, rows, &r) {
			return r
		}
		addContext(&r, actorRows, rows, len(rows)-1)
		return r
	}
	r.State = "passed"
	r.Reason = ""
	for _, step := range s.Steps {
		if step.Action == "respawn_cycle" {
			var window []Trace
			for _, row := range actorRows {
				if row.Scenario != nil && row.Scenario.StepID == step.ID {
					window = append(window, row)
				}
			}
			events := actorLifecycle(window)
			if len(events) < 2 || events[0].Kind != "actor_died" || events[1].Kind != "actor_respawned" {
				r.State = "trace_invalid"
				r.Reason = "respawn lifecycle evidence missing: " + step.ID
				r.Frame = end
				return r
			}
		}
	}
	if !checkLifecycle(s, actorRows, rows, &r) {
		return r
	}
	if r.Losses < s.Expect.ContactLosses || r.Reacquisitions < s.Expect.Reacquisitions || r.FollowResumptions < s.Expect.FollowResumptions {
		r.State = "behavior_failed"
		r.Reason = "contact cycle expectations not met"
		r.Frame = end
	}
	return r
}
