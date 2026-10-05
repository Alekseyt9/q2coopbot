package learningenv

import (
	"math"
	"testing"

	"q2coopbot/internal/policy"
)

func rewardFixture() (RewardConfig, Step, ServerOutcome) {
	id := policy.Identity{Life: 1, Map: "base1", Connection: 1, Spawncount: 42, Actor: 1, Frame: 10}
	next := policy.Observation{Identity: id, Health: 90}
	next.Identity.Frame++
	s := Step{Worker: "w", Episode: "e", Index: 1, ClientSequence: 23, Observation: policy.Observation{Identity: id, Health: 100}, Next: &next,
		Action: policy.Action{Version: policy.ActionVersion, Identity: id}, AppliedAction: policy.Action{Version: policy.ActionVersion, Identity: id},
		Execution: &Execution{Matched: true, WindowExclusive: true}, Native: &NativeStep{Spawncount: 42, Actor: 1, Sequence: 23, BeginFrame: 10, EndFrame: 11}}
	c := RewardConfig{Version: RewardVersion, MonsterDamage: .01, ReceivedDamage: -.02, SelfDamage: -.02, FriendlyDamage: -.1, Death: -5, Tick: -.001}
	o := ServerOutcome{Version: "server_step_effects_v1", Worker: "w", Episode: "e", Step: 1, Available: true, MonsterHealthDamage: 10, ReceivedHealthDamage: 10}
	return c, s, o
}

func TestRewardCostsAndFirstDeath(t *testing.T) {
	c, s, o := rewardFixture()
	o.SelfHealthDamage = 3
	o.TeammateHealthDamage = 2
	r := c.Evaluate(&s, o)
	if !r.Available || math.Abs(*r.Score-(-.361)) > 1e-10 {
		t.Fatalf("unexpected breakdown: %+v", r)
	}
	s.Terminal = true
	s.Next.Health = -10
	o.Deaths = 1
	o.ReceivedHealthDamage = 100
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["death"] != -5 {
		t.Fatalf("terminal reward: %+v", r)
	}
	// A health pickup does not erase server damage or create positive reward.
	s.Terminal = false
	s.Next.Health = 125
	o.Deaths = 0
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["received_damage"] != -2 {
		t.Fatal(r)
	}
}

func TestRewardMasksUnprovenTransitions(t *testing.T) {
	for _, name := range []string{"respawn", "tail", "gap", "handoff", "dispatch", "native", "death", "outcome", "legacy"} {
		t.Run(name, func(t *testing.T) {
			c, s, o := rewardFixture()
			switch name {
			case "respawn":
				s.Observation.Identity.Life = 2
			case "tail":
				s.Next = nil
			case "gap":
				s.Next.Identity.Frame += 2
			case "handoff":
				s.Truncated = true
			case "dispatch":
				s.Execution.WindowExclusive = false
			case "native":
				s.Native.Sequence++
			case "death":
				s.Terminal = true
				s.Next.Health = 0
			case "outcome":
				o.Episode = "other"
			case "legacy":
				o.Version = ServerOutcomeVersion
			}
			r := c.Evaluate(&s, o)
			if r.Available || r.Score != nil || r.Components != nil || r.Reason == "" {
				t.Fatal(r)
			}
		})
	}
}

func TestRewardConfigRejectsInvalidObjective(t *testing.T) {
	for _, name := range []string{"nan", "infinity", "version", "death_bonus", "no_time_cost"} {
		c, _, _ := rewardFixture()
		switch name {
		case "nan":
			c.Death = math.NaN()
		case "infinity":
			c.Tick = math.Inf(-1)
		case "version":
			c.Version = "unknown"
		case "death_bonus":
			c.Death = 1
		case "no_time_cost":
			c.Tick = 0
		}
		if c.Validate() == nil {
			t.Fatal(name)
		}
	}
}

func TestKillRewardV2LethalEvidenceAndDelayedProjectile(t *testing.T) {
	c, s, o := rewardFixture()
	c.Version, c.MonsterKill = KillRewardVersion, 5
	e := DamageEvent{Map: "base1", Spawncount: 42, Attacker: 1, Target: 67, Inflictor: 84, TargetClass: "monster_parasite", HealthBefore: 10, HealthAfter: 0, Take: 10}
	o.Events, o.MonsterKills = []DamageEvent{e}, 1
	// Current action need not fire: the projectile can have been launched earlier.
	r := c.Evaluate(&s, o)
	if !r.Available || r.Version != KillRewardVersion || r.Components["monster_kill"] != 5 || math.Abs(*r.Score-4.899) > 1e-10 {
		t.Fatal(r)
	}
	s.Truncated, s.Reason = true, "control_handoff"
	if r = c.Evaluate(&s, o); !r.Available || r.Components["monster_kill"] != 5 {
		t.Fatal("lost last verified kill at handoff", r)
	}
	s.Reason = "harness_override"
	if c.Evaluate(&s, o).Available {
		t.Fatal("accepted harness truncation")
	}
	s.Truncated, s.Reason = false, ""
	// Corpses do not earn another bonus.
	o.MonsterKills = 0
	o.Events[0].HealthBefore, o.Events[0].HealthAfter = 0, -10
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["monster_kill"] != 0 {
		t.Fatal(r)
	}
	// v1 remains byte-compatible in components, even when native effects show kills.
	c.Version, c.MonsterKill = RewardVersion, 0
	o.MonsterKills = 1
	r = c.Evaluate(&s, o)
	if _, exists := r.Components["monster_kill"]; !r.Available || exists || r.Version != RewardVersion {
		t.Fatal(r)
	}
}

func TestKillRewardRejectsUnprovenOrDuplicateKills(t *testing.T) {
	for _, name := range []string{"missing", "other_attacker", "player", "nonlethal", "duplicate", "negative", "world", "bad_health"} {
		t.Run(name, func(t *testing.T) {
			c, s, o := rewardFixture()
			c.Version, c.MonsterKill = KillRewardVersion, 5
			e := DamageEvent{Map: "base1", Spawncount: 42, Attacker: 1, Target: 67, TargetClass: "monster_parasite", HealthBefore: 10, HealthAfter: 0, Take: 10}
			o.Events, o.MonsterKills = []DamageEvent{e}, 1
			switch name {
			case "missing":
				o.Events = nil
			case "other_attacker":
				o.Events[0].Attacker = 2
			case "player":
				o.Events[0].TargetClass = "player"
			case "nonlethal":
				o.Events[0].HealthBefore, o.Events[0].HealthAfter = 20, 10
			case "duplicate":
				o.Events = append(o.Events, e)
				o.MonsterKills = 2
			case "negative":
				o.MonsterKills = -1
			case "world":
				o.Events[0].Spawncount++
			case "bad_health":
				o.Events[0].Take = 9
			}
			if r := c.Evaluate(&s, o); r.Available || r.Score != nil {
				t.Fatal(r)
			}
		})
	}
}

func TestKillRewardConfigVersionAndFiniteBonus(t *testing.T) {
	for _, bonus := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		c, _, _ := rewardFixture()
		c.Version, c.MonsterKill = KillRewardVersion, bonus
		if c.Validate() == nil {
			t.Fatal(bonus)
		}
	}
	c, _, _ := rewardFixture()
	c.MonsterKill = 5
	if c.Validate() == nil {
		t.Fatal("v1 silently changed objective")
	}
}
