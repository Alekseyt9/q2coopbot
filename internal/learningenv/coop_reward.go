package learningenv

import "fmt"

// Joint effects are offline evidence separate from this actor's damage credit.
func verifyJointDeathReward(s *Step, events []DamageEvent) (bool, error) {
	b := s.JointTerminal
	if b == nil || len(events) != len(b.DeathEventIndexes) {
		return false, fmt.Errorf("missing joint death effects")
	}
	role := -1
	for i, actor := range b.Actors {
		if actor == s.Observation.Identity.Actor {
			role = i
		}
	}
	copy := *s
	if err := b.MarkTerminal(&copy, role); err != nil {
		return false, err
	}
	seen := map[int]bool{}
	for i, e := range events {
		if e.Target != b.Actors[0] && e.Target != b.Actors[1] {
			return false, fmt.Errorf("unexpected participant death actor")
		}
		found := false
		for _, index := range s.Native.DamageIndexes {
			if index == b.DeathEventIndexes[i] {
				found = true
			}
		}
		if !found || seen[e.Target] || e.Map != b.Map || e.Spawncount != b.Spawncount || e.Frame < b.BeginFrame || e.Frame > b.EndFrame || e.TargetClass != "player" || e.HealthBefore <= 0 || e.HealthAfter > 0 || e.Take <= 0 || e.HealthBefore-e.Take != e.HealthAfter {
			return false, fmt.Errorf("invalid joint death effect")
		}
		seen[e.Target] = true
	}
	for i, actor := range b.Actors {
		if seen[actor] != (b.HealthAfter[i] <= 0) {
			return false, fmt.Errorf("joint death effect/actor mismatch")
		}
	}
	if len(seen) > 2 {
		return false, fmt.Errorf("unexpected actor in joint death effects")
	}
	return b.HealthAfter[1-role] <= 0, nil
}
