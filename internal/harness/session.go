package harness

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"
)

// Session retains a separate frame budget and behavior expectations per map.
// A transition is requested only after the preceding phase completed its steps.
type Session struct {
	ReadinessBarrier    bool    `json:"readiness_barrier,omitempty"`
	Version             int     `json:"version"`
	Name                string  `json:"name"`
	TransitionTimeoutMS int     `json:"transition_timeout_ms"`
	Phases              []Phase `json:"phases"`
}

type Phase struct {
	ID       string   `json:"id"`
	Scenario Scenario `json:"scenario"`
}

func LoadSession(path string) (Session, error) {
	var session Session
	f, err := os.Open(path)
	if err != nil {
		return session, err
	}
	defer f.Close()
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err := d.Decode(&session); err != nil {
		return session, err
	}
	if d.Decode(new(any)) != io.EOF {
		return session, fmt.Errorf("session must contain one JSON document")
	}
	return session, session.Validate()
}

func (s Session) Validate() error {
	if s.Version != 1 || s.Name == "" || len(s.Phases) < 2 || len(s.Phases) > 16 || s.TransitionTimeoutMS < 100 || s.TransitionTimeoutMS > 120000 {
		return fmt.Errorf("invalid session header")
	}
	ids := map[string]bool{}
	for _, phase := range s.Phases {
		if phase.ID == "" || ids[phase.ID] {
			return fmt.Errorf("empty or duplicate phase id %q", phase.ID)
		}
		ids[phase.ID] = true
		if err := phase.Scenario.Validate(); err != nil {
			return fmt.Errorf("phase %s: %w", phase.ID, err)
		}
		if len(phase.Scenario.Expect.MapSequence) != 0 || phase.Scenario.Expect.Failure != nil {
			return fmt.Errorf("phase %s: session phases require normal completion and use the session map order", phase.ID)
		}
	}
	return nil
}

type SessionStatus struct {
	StartFrame int           `json:"start_frame,omitempty"`
	State      string        `json:"state"`
	PhaseIndex int           `json:"phase_index"`
	PhaseID    string        `json:"phase_id"`
	Phase      Status        `json:"phase"`
	Location   FrameLocation `json:"location"`
	Reason     string        `json:"reason,omitempty"`
}

type SessionDecision struct {
	Decision
	// ChangeMap must be applied after recording the completed phase's snapshot.
	// Empty means no request. The destination is a validated simple map name.
	ChangeMap string
}

type SessionRunner struct {
	definition         Session
	Status             SessionStatus
	phase              *Runner
	lastTime           time.Duration
	clockStarted       bool
	transitionAt       time.Duration
	previousMap        string
	previousGeneration int
	seenGenerations    map[int]bool
}

func NewSession(s Session) (*SessionRunner, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return &SessionRunner{definition: s, phase: New(s.Phases[0].Scenario), Status: SessionStatus{State: "pending", PhaseID: s.Phases[0].ID}, seenGenerations: map[int]bool{}}, nil
}

func (r *SessionRunner) fail(reason string) SessionDecision {
	r.Status.State, r.Status.Reason = "failed", reason
	return SessionDecision{}
}

// Poll keeps the loading watchdog active when no new server snapshot arrives.
func (r *SessionRunner) Poll(elapsed time.Duration) {
	if r.Status.State == "completed" || r.Status.State == "failed" {
		return
	}
	if elapsed < 0 || r.clockStarted && elapsed < r.lastTime {
		r.fail("clock_reversed")
		return
	}
	r.clockStarted, r.lastTime = true, elapsed
	if r.Status.State == "waiting_map" && elapsed-r.transitionAt >= time.Duration(r.definition.TransitionTimeoutMS)*time.Millisecond {
		r.fail("map_transition_timeout")
	}
}

// Tick receives monotonic elapsed wall time separately from map-local frames.
// Call it during loading too (with the last snapshot) so a silent server times out.
func (r *SessionRunner) Tick(in Input, elapsed time.Duration) SessionDecision {
	if r.Status.State == "completed" || r.Status.State == "failed" {
		return SessionDecision{}
	}
	if elapsed < 0 || r.clockStarted && elapsed < r.lastTime {
		return r.fail("clock_reversed")
	}
	r.clockStarted, r.lastTime = true, elapsed
	r.Status.Location = FrameLocation{Map: in.Map, Generation: in.Generation, Frame: in.Frame}
	if r.Status.State == "waiting_map" {
		if elapsed-r.transitionAt >= time.Duration(r.definition.TransitionTimeoutMS)*time.Millisecond {
			return r.fail("map_transition_timeout")
		}
		next := r.definition.Phases[r.Status.PhaseIndex+1]
		if in.Generation == r.previousGeneration {
			if in.Map != r.previousMap {
				return r.fail("map_changed_without_generation")
			}
			return SessionDecision{}
		}
		if in.Map != next.Scenario.Map {
			return r.fail("unexpected_transition_map")
		}
		if r.seenGenerations[in.Generation] {
			return r.fail("generation_revisited")
		}
		r.Status.PhaseIndex++
		r.Status.PhaseID = next.ID
		r.Status.StartFrame = 0
		r.Status.State = "pending"
		r.phase = New(next.Scenario)
		r.seenGenerations[in.Generation] = true
		r.previousMap, r.previousGeneration = in.Map, in.Generation
	}
	// Once a phase's generation has been observed, changes before StartFrame
	// must not be silently accepted by the single-map runner's pending state.
	if len(r.seenGenerations) > 0 && (in.Map != r.previousMap || in.Generation != r.previousGeneration) {
		return r.fail("unexpected_phase_generation")
	}
	if len(r.seenGenerations) == 0 && in.Map == r.definition.Phases[0].Scenario.Map {
		r.seenGenerations[in.Generation] = true
		r.previousMap, r.previousGeneration = in.Map, in.Generation
	}
	if r.definition.ReadinessBarrier && !r.phase.started {
		if in.PhaseStart == 0 {
			r.Status.State = "pending"
			r.Status.Phase = r.phase.Status
			return SessionDecision{}
		}
		if r.Status.StartFrame == 0 {
			if in.PhaseStart < in.Frame {
				return r.fail("barrier_start_missed")
			}
			delta := in.PhaseStart - r.phase.Scenario.StartFrame
			r.phase.Scenario.StartFrame = in.PhaseStart
			r.phase.Scenario.GameFrames += delta
			r.Status.StartFrame = in.PhaseStart
		} else if r.Status.StartFrame != in.PhaseStart {
			return r.fail("barrier_start_changed")
		}
	}
	d := r.phase.Tick(in)
	r.Status.Phase = r.phase.Status
	r.Status.State = r.phase.Status.State
	if r.phase.Status.State == "failed" {
		return r.fail(r.phase.Status.Reason)
	}
	if r.phase.Status.State != "completed" {
		return SessionDecision{Decision: d}
	}
	if r.Status.PhaseIndex == len(r.definition.Phases)-1 {
		return SessionDecision{}
	}
	r.Status.State = "waiting_map"
	r.transitionAt = elapsed
	return SessionDecision{ChangeMap: r.definition.Phases[r.Status.PhaseIndex+1].Scenario.Map}
}

func (r *SessionRunner) RejectRoute(reason string) {
	if r.Status.State != "running" {
		return
	}
	r.phase.RejectRoute(reason)
	r.Status.Phase = r.phase.Status
	if r.phase.Status.State == "failed" {
		r.fail(r.phase.Status.Reason)
	}
}
