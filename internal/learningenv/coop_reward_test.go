package learningenv

import (
	"math"
	"testing"
)

func coopRewardFixture() (RewardConfig, Step, ServerOutcome) {
	c, s, o := qualityFixture()
	c.Version, c.NavigationPotential, c.PeerDeath = CoopRewardVersion, .1, -5
	s.Next.Health = 100
	o.MonsterHealthDamage, o.ReceivedHealthDamage = 0, 0
	s.Native.DamageIndexes = []int{5}
	b := &PairedDeathBoundary{Version: "coop_joint_death_boundary_v1", Reason: "coop_participant_death", Map: "base1", Spawncount: 42, Seed: 123, BeginFrame: 10, EndFrame: 11, Actors: [2]int{1, 2}, HealthBefore: [2]int16{100, 4}, HealthAfter: [2]int16{100, 0}, DeathEventIndexes: []int{5}}
	if err := b.MarkTerminal(&s, 0); err != nil {
		panic(err)
	}
	o.JointDeathEvents = []DamageEvent{{Map: "base1", Spawncount: 42, Frame: 11, Target: 2, TargetClass: "player", HealthBefore: 4, HealthAfter: 0, Take: 4}}
	return c, s, o
}

func TestCoopRewardPeerDeathAndOwnDeath(t *testing.T) {
	c, s, o := coopRewardFixture()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	r := c.Evaluate(&s, o)
	if !r.Available || r.Components["peer_death"] != -5 || r.Components["death"] != 0 || r.Components["monster_damage"] != 0 || r.Components["monster_kill"] != 0 {
		t.Fatal(r)
	}
	s.Next.Health = 0
	s.JointTerminal.HealthAfter = [2]int16{0, 100}
	s.JointTerminal.HealthBefore[1] = 100
	o.JointDeathEvents[0].Target = 1
	o.JointDeathEvents[0].HealthBefore = 100
	o.JointDeathEvents[0].Take = 100
	o.Deaths = 1
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["death"] != -5 || r.Components["peer_death"] != 0 {
		t.Fatal(r)
	}
	// Simultaneous loss uses the own-death cost once, without stacking peer cost.
	s.JointTerminal.HealthBefore[1] = 4
	s.JointTerminal.HealthAfter[1] = 0
	s.JointTerminal.DeathEventIndexes = append(s.JointTerminal.DeathEventIndexes, 6)
	s.Native.DamageIndexes = append(s.Native.DamageIndexes, 6)
	o.JointDeathEvents = append(o.JointDeathEvents, DamageEvent{Map: "base1", Spawncount: 42, Frame: 11, Target: 2, TargetClass: "player", HealthBefore: 4, HealthAfter: 0, Take: 4})
	r = c.Evaluate(&s, o)
	if !r.Available || r.Components["death"] != -5 || r.Components["peer_death"] != 0 {
		t.Fatal(r)
	}
}

func TestCoopRewardRejectsUnprovenJointDeath(t *testing.T) {
	for _, name := range []string{"missing_effect", "wrong_actor", "wrong_frame", "bad_damage", "missing_index", "alive_peer", "wrong_generation", "not_terminal", "legacy"} {
		t.Run(name, func(t *testing.T) {
			c, s, o := coopRewardFixture()
			switch name {
			case "missing_effect":
				o.JointDeathEvents = nil
			case "wrong_actor":
				o.JointDeathEvents[0].Target = 3
			case "wrong_frame":
				o.JointDeathEvents[0].Frame = 12
			case "bad_damage":
				o.JointDeathEvents[0].Take = 3
			case "missing_index":
				s.Native.DamageIndexes = nil
			case "alive_peer":
				s.JointTerminal.HealthAfter[1] = 1
			case "wrong_generation":
				o.JointDeathEvents[0].Spawncount++
			case "not_terminal":
				s.Terminal = false
			case "legacy":
				c.Version = NavigationRewardVersion
				c.PeerDeath = 0
			}
			if r := c.Evaluate(&s, o); r.Available {
				t.Fatal("unproven joint reward accepted", r)
			}
		})
	}
}

func TestCoopRewardConfigAndNonterminal(t *testing.T) {
	c, s, o := coopRewardFixture()
	s.JointTerminal = nil
	s.Terminal = false
	s.Reason = ""
	o.JointDeathEvents = nil
	r := c.Evaluate(&s, o)
	if !r.Available || r.Components["peer_death"] != 0 {
		t.Fatal(r)
	}
	for _, value := range []float64{0, 1, -10.1, math.NaN(), math.Inf(1)} {
		bad := c
		bad.PeerDeath = value
		if bad.Validate() == nil {
			t.Fatal("invalid peer cost accepted", value)
		}
	}
	c.Version = NavigationRewardVersion
	if c.Validate() == nil {
		t.Fatal("legacy accepted peer cost")
	}
}
