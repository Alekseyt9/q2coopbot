package harness

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"

	"q2coopbot/internal/quake"
)

type Step struct {
	ID      string      `json:"id"`
	Action  string      `json:"action"`
	Frames  int         `json:"frames,omitempty"`
	Timeout int         `json:"timeout_frames,omitempty"`
	Target  *quake.Vec3 `json:"target,omitempty"`
	Route   bool        `json:"route,omitempty"`
}
type Scenario struct {
	BotReleaseFrame int          `json:"bot_release_frame,omitempty"`
	ActorHealth     int          `json:"actor_health,omitempty"`
	BotInvulnerable bool         `json:"bot_invulnerable,omitempty"`
	BotHealth       int          `json:"bot_health,omitempty"`
	MapEntry        string       `json:"map_entry,omitempty"`
	Expect          Expectations `json:"expect"`
	Version         int          `json:"version"`
	Name            string       `json:"name"`
	Map             string       `json:"map"`
	StartFrame      int          `json:"start_frame"`
	GameFrames      int          `json:"game_frames"`
	ActorOrigin     quake.Vec3   `json:"actor_origin"`
	BotOrigin       quake.Vec3   `json:"bot_origin"`
	Steps           []Step       `json:"steps"`
}
type Expectations struct {
	MapSequence       []string         `json:"map_sequence,omitempty"`
	Failure           *ExpectedFailure `json:"failure,omitempty"`
	Invariants        []string         `json:"invariants,omitempty"`
	ContactLosses     int              `json:"contact_losses"`
	Reacquisitions    int              `json:"reacquisitions"`
	FollowResumptions int              `json:"follow_resumptions"`
}

type ExpectedFailure struct {
	MinStallFrames int    `json:"min_stall_frames,omitempty"`
	NearbyKind     string `json:"nearby_kind,omitempty"`
	Reason         string `json:"reason"`
	StepID         string `json:"step_id"`
	MinFrame       int    `json:"min_frame"`
	MaxFrame       int    `json:"max_frame"`
}

func Load(path string) (Scenario, error) {
	var s Scenario
	f, err := os.Open(path)
	if err != nil {
		return s, err
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err = d.Decode(&s); err != nil {
		return s, err
	}
	if err = d.Decode(new(any)); err != io.EOF {
		return s, fmt.Errorf("scenario must contain one JSON document")
	}
	return s, s.Validate()
}

func (s Scenario) Validate() error {
	if s.BotReleaseFrame != 0 && (s.BotReleaseFrame <= s.StartFrame || s.BotReleaseFrame >= s.GameFrames) {
		return fmt.Errorf("bot_release_frame must be between start_frame and game_frames")
	}
	if s.ActorHealth < 0 || s.ActorHealth > 100 {
		return fmt.Errorf("actor_health must be 0 (default) or 1..100")
	}
	if s.BotHealth < 0 || s.BotHealth > 100 {
		return fmt.Errorf("bot_health must be 0 (default) or 1..100")
	}
	if s.MapEntry != "" && !regexp.MustCompile(`^[a-zA-Z0-9_]+$`).MatchString(s.MapEntry) {
		return fmt.Errorf("invalid map_entry")
	}
	if len(s.Expect.MapSequence) > 0 {
		if len(s.Expect.MapSequence) < 2 || len(s.Expect.MapSequence) > 16 || s.Expect.MapSequence[len(s.Expect.MapSequence)-1] != s.Map {
			return fmt.Errorf("map_sequence requires 2..16 maps ending at scenario map")
		}
		for _, name := range s.Expect.MapSequence {
			if !regexp.MustCompile(`^[a-zA-Z0-9_]+$`).MatchString(name) {
				return fmt.Errorf("invalid map_sequence map %q", name)
			}
		}
	}
	seenChecks := map[string]bool{}
	for _, name := range s.Expect.Invariants {
		if !knownInvariant(name) || seenChecks[name] {
			return fmt.Errorf("unknown or duplicate invariant %q", name)
		}
		seenChecks[name] = true
	}
	if s.StartFrame > s.GameFrames {
		return fmt.Errorf("start exceeds game_frames")
	}
	if s.Expect.ContactLosses < 0 || s.Expect.Reacquisitions < 0 || s.Expect.FollowResumptions < 0 {
		return fmt.Errorf("negative expectation")
	}
	if s.Version != 1 || s.Name == "" || !regexp.MustCompile(`^[a-zA-Z0-9_]+$`).MatchString(s.Map) || s.StartFrame < 40 || s.GameFrames < 100 || s.GameFrames > 10000 || len(s.Steps) == 0 || len(s.Steps) > 64 {
		return fmt.Errorf("invalid scenario header")
	}
	finite := func(v quake.Vec3) bool {
		for _, x := range v {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				return false
			}
		}
		return true
	}
	if !finite(s.ActorOrigin) || !finite(s.BotOrigin) {
		return fmt.Errorf("invalid initial position")
	}
	ids := map[string]bool{}
	budget := s.StartFrame
	for _, step := range s.Steps {
		if step.ID == "" || ids[step.ID] {
			return fmt.Errorf("duplicate or empty step id %q", step.ID)
		}
		ids[step.ID] = true
		switch step.Action {
		case "push":
			if step.Target == nil || !finite(*step.Target) || step.Frames < 1 || step.Frames > 100 || step.Timeout != 0 || step.Route {
				return fmt.Errorf("invalid push step %s", step.ID)
			}
			budget += step.Frames + 1
		case "respawn_cycle":
			if step.Timeout < 1 || step.Timeout > 1000 || step.Target != nil || step.Frames != 0 || step.Route {
				return fmt.Errorf("invalid respawn cycle %s", step.ID)
			}
			budget += step.Timeout + 1
		case "wait":
			if step.Frames < 1 || step.Frames > 1000 || step.Timeout != 0 || step.Target != nil || step.Route {
				return fmt.Errorf("invalid wait step %s", step.ID)
			}
			budget += step.Frames + 1
		case "walk", "place":
			if step.Target == nil || !finite(*step.Target) || step.Timeout < 1 || step.Timeout > 1000 || step.Frames != 0 || step.Action == "place" && step.Route {
				return fmt.Errorf("invalid movement step %s", step.ID)
			}
			budget += step.Timeout + 1
		default:
			return fmt.Errorf("unknown action %q", step.Action)
		}
	}
	if budget > s.GameFrames {
		return fmt.Errorf("scenario deadlines exceed game_frames")
	}
	if f := s.Expect.Failure; f != nil {
		if f.MinStallFrames != 0 && (f.MinStallFrames < 3 || f.MinStallFrames > s.GameFrames) || f.NearbyKind != "" && f.NearbyKind != "teammate" && f.NearbyKind != "enemy" || f.NearbyKind != "" && f.MinStallFrames < 3 {
			return fmt.Errorf("invalid expected stall evidence")
		}
		if f.MinStallFrames > 0 && f.Reason != "step_timeout" {
			return fmt.Errorf("stall evidence requires timeout")
		}
		if f.Reason != "step_timeout" && !routeFailure(f.Reason) || !ids[f.StepID] || f.MinFrame < s.StartFrame || f.MaxFrame < f.MinFrame || f.MaxFrame > s.GameFrames {
			return fmt.Errorf("invalid expected failure")
		}
		for _, step := range s.Steps {
			if step.ID == f.StepID && routeFailure(f.Reason) && (step.Action != "walk" || !step.Route) {
				return fmt.Errorf("route failure requires routed walking")
			}
			if step.ID == f.StepID && step.Timeout == 0 {
				return fmt.Errorf("expected timeout requires a timed step")
			}
		}
		if s.Expect.ContactLosses != 0 || s.Expect.Reacquisitions != 0 || s.Expect.FollowResumptions != 0 {
			return fmt.Errorf("expected failure cannot include contact cycle expectations")
		}
	}
	return nil
}

type Status struct {
	DeathFrame     int    `json:"death_frame,omitempty"`
	RespawnFrame   int    `json:"respawn_frame,omitempty"`
	MovementReason string `json:"movement_reason,omitempty"`
	State          string `json:"state"`
	StepID         string `json:"step_id,omitempty"`
	StepIndex      int    `json:"step_index"`
	StepStart      int    `json:"step_start_frame"`
	CompletedSteps int    `json:"completed_steps"`
	EndFrame       int    `json:"end_frame,omitempty"`
	Reason         string `json:"reason,omitempty"`
}

func routeFailure(reason string) bool {
	switch reason {
	case "aas_unavailable", "route_target_outside_aas", "route_start_outside_aas", "route_unavailable":
		return true
	}
	return false
}

// RejectRoute records a failed preflight at the current step before movement.
func (r *Runner) RejectRoute(reason string) {
	if r.Status.State == "running" && routeFailure(reason) {
		r.fail(r.lastFrame, reason)
	}
}

type Input struct {
	Connection        int
	PhaseStart        int
	Frame, Generation int
	Map               string
	Self              quake.Vec3
	OnGround          bool
	Health            int16
}
type Decision struct {
	Push          *quake.Vec3
	Kill, Respawn bool
	Place, Walk   *quake.Vec3
	Route         bool
	NewStep       bool
}
type Runner struct {
	Scenario              Scenario
	Status                Status
	started, enter        bool
	generation, lastFrame int
}

func New(s Scenario) *Runner {
	return &Runner{Scenario: s, Status: Status{State: "pending"}, enter: true, lastFrame: -1}
}
func (r *Runner) fail(frame int, reason string) Decision {
	r.Status.State = "failed"
	r.Status.EndFrame = frame
	r.Status.Reason = reason
	return Decision{}
}
func (r *Runner) Tick(in Input) Decision {
	if r.Status.State == "failed" || r.Status.State == "completed" {
		return Decision{}
	}
	if !r.started {
		if in.Map != r.Scenario.Map || in.Frame < r.Scenario.StartFrame {
			return Decision{}
		}
		if in.Frame != r.Scenario.StartFrame {
			return r.fail(in.Frame, "late_start")
		}
		r.started = true
		r.generation = in.Generation
		r.Status.State = "running"
	} else {
		if in.Map != r.Scenario.Map || in.Generation != r.generation {
			return r.fail(in.Frame, "map_generation_changed")
		}
		if in.Frame == r.lastFrame {
			return Decision{}
		}
		if in.Frame != r.lastFrame+1 {
			return r.fail(in.Frame, "frame_discontinuity")
		}
	}
	r.lastFrame = in.Frame
	if r.enter {
		r.Status.StepIndex = r.Status.CompletedSteps
	}
	step := r.Scenario.Steps[r.Status.StepIndex]
	if in.Health <= 0 && (step.Action != "respawn_cycle" || r.enter) {
		return r.fail(in.Frame, "actor_dead")
	}
	d := Decision{}
	if r.enter {
		r.enter = false
		r.Status.StepID = step.ID
		r.Status.StepStart = in.Frame
		d.NewStep = true
		if step.Action == "place" {
			d.Place = step.Target
		}
		r.Status.DeathFrame = 0
		r.Status.RespawnFrame = 0
		if step.Action == "respawn_cycle" {
			d.Kill = true
		}
	}
	elapsed := in.Frame - r.Status.StepStart
	done := (step.Action == "wait" || step.Action == "push") && elapsed >= step.Frames
	if step.Action == "respawn_cycle" {
		if in.Health <= 0 {
			if r.Status.DeathFrame == 0 {
				r.Status.DeathFrame = in.Frame
			}
			d.Respawn = (in.Frame-r.Status.DeathFrame)%2 == 0
		} else if r.Status.DeathFrame > 0 {
			r.Status.RespawnFrame = in.Frame
			done = true
		}
	}
	if (step.Action == "walk" || step.Action == "place") && step.Target != nil && elapsed > 0 {
		distance := math.Sqrt(math.Pow(in.Self[0]-step.Target[0], 2) + math.Pow(in.Self[1]-step.Target[1], 2) + math.Pow(in.Self[2]-step.Target[2], 2))
		done = distance <= 16 && (step.Action == "place" || in.OnGround)
	}
	if done {
		r.Status.CompletedSteps++
		if r.Status.CompletedSteps == len(r.Scenario.Steps) {
			r.Status.State = "completed"
			r.Status.EndFrame = in.Frame
		} else {
			r.enter = true
		}
		return Decision{}
	}
	if step.Timeout > 0 && elapsed >= step.Timeout {
		return r.fail(in.Frame, "step_timeout")
	}
	if step.Action == "walk" {
		d.Walk = step.Target
		d.Route = step.Route
	}
	if step.Action == "push" {
		d.Push = step.Target
	}
	return d
}
