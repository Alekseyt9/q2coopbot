package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// RunnerCheckpoint preserves progress of reversible wait/walk steps. Binding
// this payload and the bot payload to a native save needs a coordinated barrier.
type RunnerCheckpoint struct {
	Version        int    `json:"version"`
	ScenarioSHA256 string `json:"scenario_sha256"`
	CapturedFrame  int    `json:"captured_frame"`
	StepIndex      int    `json:"step_index"`
	StepID         string `json:"step_id"`
	Elapsed        int    `json:"elapsed_frames"`
}

func scenarioDigest(s Scenario) string {
	data, _ := json.Marshal(s)
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

func (r *Runner) CaptureCheckpoint() (RunnerCheckpoint, error) {
	state := RunnerCheckpoint{Version: 1, ScenarioSHA256: scenarioDigest(r.Scenario), CapturedFrame: r.lastFrame, StepIndex: r.Status.StepIndex, StepID: r.Status.StepID, Elapsed: r.lastFrame - r.Status.StepStart}
	if !r.started || r.enter || r.Status.State != "running" {
		return state, fmt.Errorf("runner checkpoint requires an active reversible step")
	}
	return state, state.validate(r.Scenario)
}

func (state RunnerCheckpoint) validate(s Scenario) error {
	if state.Version != 1 || state.ScenarioSHA256 != scenarioDigest(s) || state.CapturedFrame < 0 || state.Elapsed < 0 || state.Elapsed > state.CapturedFrame || state.StepIndex < 0 || state.StepIndex >= len(s.Steps) {
		return fmt.Errorf("runner checkpoint identity/progress mismatch")
	}
	step := s.Steps[state.StepIndex]
	if step.ID != state.StepID || step.Action != "wait" && step.Action != "walk" || step.Action == "wait" && state.Elapsed >= step.Frames || step.Action == "walk" && state.Elapsed >= step.Timeout {
		return fmt.Errorf("runner checkpoint step cannot be resumed")
	}
	return nil
}

func RestoreRunnerCheckpoint(s Scenario, state RunnerCheckpoint, fresh Input) (*Runner, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if err := state.validate(s); err != nil {
		return nil, err
	}
	if fresh.Map != s.Map || fresh.Frame < 1 || fresh.Generation < 0 || fresh.Health <= 0 || !fresh.OnGround {
		return nil, fmt.Errorf("runner restore needs a fresh living grounded actor")
	}
	r := New(s)
	r.started = true
	r.enter = false
	r.generation = fresh.Generation
	r.lastFrame = fresh.Frame - 1
	r.Status = Status{State: "running", StepIndex: state.StepIndex, StepID: state.StepID, StepStart: fresh.Frame - state.Elapsed, CompletedSteps: state.StepIndex}
	return r, nil
}
