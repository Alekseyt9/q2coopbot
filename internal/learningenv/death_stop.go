package learningenv

import (
	"fmt"

	"q2coopbot/internal/policy"
)

// DeathStop is supervisor evidence, never an additional policy observation.
type DeathStop struct {
	Version       string `json:"version"`
	Reason        string `json:"reason"`
	Map           string `json:"map"`
	Spawncount    int    `json:"spawncount"`
	Actor         int    `json:"actor"`
	DeathFrame    int    `json:"death_frame"`
	ObservedFrame int    `json:"observed_frame"`
	Health        int    `json:"health"`
}

// VerifyDeathStop requires the complete terminal and its ordered native effect.
// It does not create a terminal from a receipt or a stale health observation.
func VerifyDeathStop(g DeathStop, release *CombatRelease, events []DamageEvent, s *Step) error {
	fail := func() error { return fmt.Errorf("unverified first-life death stop") }
	if s == nil || release == nil || s.Next == nil || s.Native == nil || s.Execution == nil ||
		!s.Execution.Matched || !s.Execution.WindowExclusive || s.Execution.RecoveryCommands != 0 || !s.Terminal || s.Truncated || s.Reason != "observed_death" {
		return fail()
	}
	id, next, n := s.Observation.Identity, s.Next.Identity, s.Native
	if g.Version != "combat_first_life_death_stop_v1" || g.Reason != "combat_first_life_death" ||
		(g.Map != "base1" && g.Map != "base2") || g.Map != id.Map || id.Life != 1 || id.Connection != 1 ||
		!policy.SameLife(id, next) || next.Frame != id.Frame+1 || s.Observation.Health <= 0 || s.Next.Health > 0 ||
		g.Health > 0 || g.Actor <= 0 || g.Actor != id.Actor || g.Spawncount != id.Spawncount ||
		release.Spawncount != id.Spawncount || id.Frame < release.Frame ||
		g.DeathFrame < release.Frame || g.ObservedFrame < next.Frame ||
		(g.ObservedFrame == next.Frame && g.Health != int(s.Next.Health)) ||
		n.Spawncount != id.Spawncount || n.Actor != id.Actor || n.BeginFrame != id.Frame || n.EndFrame != next.Frame || n.Sequence == 0 || n.Sequence != s.ClientSequence {
		return fail()
	}
	first := -1
	for i, e := range events {
		if e.Spawncount == id.Spawncount && e.Map == id.Map && e.Target == id.Actor && e.TargetClass == "player" &&
			e.Frame >= release.Frame && e.HealthBefore > 0 && e.HealthAfter <= 0 {
			first = i
			break
		}
	}
	if first < 0 || events[first].Frame != g.DeathFrame || g.DeathFrame < n.BeginFrame || g.DeathFrame > n.EndFrame {
		return fail()
	}
	for _, index := range n.DamageIndexes {
		if index == first {
			return nil
		}
	}
	return fail()
}
