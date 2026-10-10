package learningenv

import (
	"fmt"
	"q2coopbot/internal/harness"
	"q2coopbot/internal/policy"
)

// PairedDeathBoundary is offline supervisor evidence, never policy input.
// It closes both roles at the same completed native tick, including effects.
type PairedDeathBoundary struct {
	Version           string   `json:"version"`
	Reason            string   `json:"reason"`
	Map               string   `json:"map"`
	Spawncount        int      `json:"spawncount"`
	Seed              int      `json:"seed"`
	BeginFrame        int      `json:"begin_frame"`
	EndFrame          int      `json:"end_frame"`
	Actors            [2]int   `json:"actors"`
	HealthBefore      [2]int16 `json:"health_before"`
	HealthAfter       [2]int16 `json:"health_after"`
	DeathEventIndexes []int    `json:"death_event_indexes"`
}

func (b *PairedDeathBoundary) MarkTerminal(s *Step, role int) error {
	if b == nil || role < 0 || role > 1 || s == nil || s.Next == nil || s.Native == nil || s.Execution == nil || !s.Execution.Matched || !s.Execution.WindowExclusive {
		return fmt.Errorf("unproven paired terminal transition")
	}
	if b.Version != "coop_joint_death_boundary_v1" || b.Reason != "coop_participant_death" || len(b.DeathEventIndexes) == 0 || b.Actors[0] <= 0 || b.Actors[1] <= 0 || b.Actors[0] == b.Actors[1] || b.HealthAfter[0] > 0 && b.HealthAfter[1] > 0 {
		return fmt.Errorf("invalid paired death boundary")
	}
	id, next := s.Observation.Identity, s.Next.Identity
	if !policy.SameLife(id, next) || id.Life != 1 || id.Connection != 1 || id.Map != b.Map || id.Spawncount != b.Spawncount || id.Actor != b.Actors[role] || id.Frame != b.BeginFrame || next.Frame != b.EndFrame || b.EndFrame != b.BeginFrame+1 || s.Observation.Health != b.HealthBefore[role] || s.Next.Health != b.HealthAfter[role] || s.Native.Actor != id.Actor || s.Native.BeginFrame != b.BeginFrame || s.Native.EndFrame != b.EndFrame || s.Native.Spawncount != b.Spawncount || s.Native.Sequence != s.ClientSequence {
		return fmt.Errorf("paired terminal disagrees with selected participant")
	}
	s.Terminal, s.Truncated, s.Reason, s.JointTerminal = true, false, "coop_participant_death", b
	return nil
}

func (n *NativePairs) FirstDeathBoundary(traces [2][]harness.Trace, events []DamageEvent) (*PairedDeathBoundary, error) {
	if _, _, err := n.BindTraces(traces, 0); err != nil {
		return nil, err
	}
	for i, pair := range n.Pairs {
		step := pair.Steps[0]
		if step.BeginFrame < n.Release.Frame {
			continue
		}
		for role := range traces {
			if traces[role][i].Health == nil || *traces[role][i].Health <= 0 {
				return nil, fmt.Errorf("paired participant already dead or health unknown before shared boundary")
			}
		}
		var deaths []int
		for _, index := range step.DamageIndexes {
			if index < 0 || index >= len(events) {
				return nil, fmt.Errorf("paired boundary invalid damage index")
			}
			e := events[index]
			if e.TargetClass != "player" || e.HealthBefore <= 0 || e.HealthAfter > 0 {
				continue
			}
			if e.Target != pair.Steps[0].Actor && e.Target != pair.Steps[1].Actor {
				continue
			}
			if e.Spawncount != step.Spawncount || e.Frame < step.BeginFrame || e.Frame > step.EndFrame {
				return nil, fmt.Errorf("paired death outside shared tick")
			}
			deaths = append(deaths, index)
		}
		if len(deaths) == 0 {
			continue
		}
		if i+1 >= len(traces[0]) || i+1 >= len(traces[1]) {
			return nil, fmt.Errorf("paired death lacks both next observations")
		}
		b := &PairedDeathBoundary{Version: "coop_joint_death_boundary_v1", Reason: "coop_participant_death", Map: traces[0][i].Map, Spawncount: step.Spawncount, Seed: n.Release.Seed, BeginFrame: step.BeginFrame, EndFrame: step.EndFrame, DeathEventIndexes: deaths}
		for role := range traces {
			before, after := traces[role][i], traces[role][i+1]
			if after.Frame != step.EndFrame || before.Map != b.Map || after.Map != b.Map || after.Health == nil {
				return nil, fmt.Errorf("paired death next observation missing or changed map")
			}
			b.Actors[role], b.HealthBefore[role], b.HealthAfter[role] = pair.Steps[role].Actor, *before.Health, *after.Health
			dead := false
			for _, index := range deaths {
				if events[index].Target == b.Actors[role] {
					dead = true
				}
			}
			if dead != (*after.Health <= 0) {
				return nil, fmt.Errorf("paired death effect disagrees with participant observation")
			}
		}
		return b, nil
	}
	return nil, nil
}
