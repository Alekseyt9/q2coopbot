// Package learningenv assembles client-observed steps without importing server
// ground truth into policy inputs. It does not yet run an optimizer or PPO.
package learningenv

import (
	"fmt"
	"strings"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

const StepVersion = "combat_step_v1"
const OutcomeVersion = "observed_outcome_v1"

type Step struct {
	JointTerminal  *PairedDeathBoundary  `json:"joint_terminal,omitempty"`
	Sample         *policy.Sample        `json:"sample,omitempty"`
	Native         *NativeStep           `json:"native_step,omitempty"`
	ClientSequence uint32                `json:"client_sequence"`
	Execution      *Execution            `json:"server_execution"`
	Version        string                `json:"version"`
	Worker         string                `json:"worker"`
	Episode        string                `json:"episode"`
	Index          int                   `json:"index"`
	Observation    policy.Observation    `json:"observation"`
	Action         policy.Action         `json:"action"`
	AppliedAction  policy.Action         `json:"applied_action"`
	LabelQuality   string                `json:"label_quality"`
	Command        quake.UserCmd         `json:"sent_command"`
	Next           *policy.Observation   `json:"next_observation"`
	Owner          string                `json:"owner"`
	Provider       string                `json:"provider"`
	Interventions  []policy.Intervention `json:"interventions,omitempty"`
	Terminal       bool                  `json:"terminal"`
	Truncated      bool                  `json:"truncated"`
	Reason         string                `json:"reason,omitempty"`
}

// Outcomes are emitted to a different file; no scalar reward or victory is
// invented from disappearance of a target. Health changes are observed deltas,
// not true damage attribution (pickups may mask damage).
type Outcome struct {
	Version       string   `json:"version"`
	Worker        string   `json:"worker"`
	Episode       string   `json:"episode"`
	Step          int      `json:"step"`
	HealthDelta   *int     `json:"health_delta"`
	ArmorDelta    *int     `json:"armor_delta"`
	ObservedDeath bool     `json:"observed_death"`
	Score         *float64 `json:"reward_score"`
}

type Assembler struct {
	Worker, Episode string
	// Diagnostic paired peer only: retain scripted idle/walking after verified
	// release. Such datasets remain explicitly ineligible for PPO.
	KeepScriptedPeerFromFrame int
	pending                   *policy.Capture
	index                     int
}

func (a *Assembler) harnessOverride(c policy.Capture) bool {
	if a.KeepScriptedPeerFromFrame > 0 && c.Observation.Identity.Frame >= a.KeepScriptedPeerFromFrame {
		allowed := func(reason string) bool {
			return reason == "" || reason == "test_idle" || reason == "test_walk" || !strings.HasPrefix(reason, "test_")
		}
		if allowed(c.LimitReason) && allowed(c.MoveLimitReason) {
			return false
		}
	}
	return strings.HasPrefix(c.LimitReason, "test_") || strings.HasPrefix(c.MoveLimitReason, "test_")
}

func (a *Assembler) Push(c policy.Capture) (*Step, *Outcome, error) {
	o := c.Observation
	if o.Version != policy.ObservationVersion || c.Applied.Version != policy.ActionVersion || o.Identity != c.Applied.Identity || o.Identity.Life < 1 || o.Identity.Frame < 1 {
		return nil, nil, fmt.Errorf("invalid capture identity/version/life")
	}
	var step *Step
	var outcome *Outcome
	if a.pending != nil {
		p := a.pending
		if policy.SameLife(p.Observation.Identity, o.Identity) && o.Identity.Frame <= p.Observation.Identity.Frame {
			return nil, nil, fmt.Errorf("nonmonotonic capture")
		}
		step, outcome = a.finish("", &o)
		switch {
		case a.harnessOverride(c):
			step.Truncated = true
			step.Reason = "harness_override"
			step.Next = nil
		case !policy.SameLife(p.Observation.Identity, o.Identity):
			step.Truncated = true
			step.Reason = "life_or_world_changed"
			step.Next = nil
		case o.Identity.Frame != p.Observation.Identity.Frame+1:
			step.Truncated = true
			step.Reason = "frame_gap"
			step.Next = nil
		case o.PreviousCommand != p.AppliedCommand:
			step.Truncated = true
			step.Reason = "command_history_mismatch"
			step.Next = nil
		case o.AgeMS < 0 || o.AgeMS > 300:
			step.Truncated = true
			step.Reason = "stale_next_observation"
			step.Next = nil
		case o.Health <= 0:
			step.Terminal = true
			step.Reason = "observed_death"
			outcome.ObservedDeath = true
		case owner(c) != owner(*p) || c.Provider != p.Provider:
			step.Truncated = true
			step.Reason = "control_handoff"
		}
		if step.Next != nil {
			health, armor := int(o.Health-p.Observation.Health), int(o.Armor-p.Observation.Armor)
			outcome.HealthDelta, outcome.ArmorDelta = &health, &armor
		}
	}
	a.pending = nil
	if o.Health > 0 && o.AgeMS >= 0 && o.AgeMS <= 300 && !a.harnessOverride(c) {
		copy := c
		a.pending = &copy
	}
	return step, outcome, nil
}

func owner(c policy.Capture) string {
	if c.Selection != nil {
		return c.Selection.Owner
	}
	return c.Provider
}

func (a *Assembler) finish(reason string, next *policy.Observation) (*Step, *Outcome) {
	p := a.pending
	if p == nil {
		return nil, nil
	}
	a.index++
	s := &Step{Version: StepVersion, Worker: a.Worker, Episode: a.Episode, Index: a.index, Observation: p.Observation,
		ClientSequence: p.ClientSequence,
		Action:         p.Proposed, AppliedAction: p.Applied, LabelQuality: p.LabelQuality, Command: p.AppliedCommand, Next: next, Owner: owner(*p), Provider: p.Provider, Reason: reason, Truncated: reason != ""}
	if p.Selection != nil {
		s.Sample = p.Selection.Sample
		s.Interventions = p.Selection.Interventions
	}
	o := &Outcome{Version: OutcomeVersion, Worker: a.Worker, Episode: a.Episode, Step: a.index}
	return s, o
}

func (a *Assembler) Close(reason string) (*Step, *Outcome) {
	if reason == "" {
		reason = "trace_end"
	}
	s, o := a.finish(reason, nil)
	a.pending = nil
	return s, o
}
