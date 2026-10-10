package learningenv

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func missLog(outcome string) string {
	return "g_test_projectile ready version=1\n" +
		"sv_test_step version=1 phase=begin spawncount=42 frame=9 seq=22 actor=1\n" +
		"sv_test_projectile spawncount=42 server_frame=9 g_test_projectile version=1 event=spawn map=base1 frame=11 shot=1 entity=350 attacker=1 mod=1 x=0 y=0 z=0 vx=1000 vy=0 vz=0\n" +
		"sv_test_step version=1 phase=end spawncount=42 frame=10 seq=22 actor=1\n" +
		"sv_test_step version=1 phase=begin spawncount=42 frame=10 seq=23 actor=1\n" +
		fmt.Sprintf("sv_test_projectile spawncount=42 server_frame=10 g_test_projectile version=1 event=end map=base1 frame=12 shot=1 entity=350 target=0 outcome=%s\n", outcome) +
		"sv_test_step version=1 phase=end spawncount=42 frame=11 seq=23 actor=1\n"
}

func TestProjectileMissDelayedAndExactlyOnce(t *testing.T) {
	j, e := ReadProjectileMisses(strings.NewReader(missLog("geometry")), nil)
	if e != nil {
		t.Fatal(e)
	}
	_, s, o := rewardFixture()
	first := s
	first.Native = &NativeStep{Spawncount: 42, BeginFrame: 9, EndFrame: 10, Sequence: 22, Actor: 1}
	if e = j.Enrich(&first, &o); e != nil || len(o.ProjectileMisses.Misses) != 0 {
		t.Fatal("flying projectile penalized", e, o)
	}
	if e = j.Enrich(&s, &o); e != nil || len(o.ProjectileMisses.Misses) != 1 {
		t.Fatal("delayed miss missing", e, o)
	}
	if e = j.Enrich(&s, &o); e == nil {
		t.Fatal("duplicate ending accepted")
	}
}

func TestProjectileMissHitUnknownAndLifeMasks(t *testing.T) {
	for _, outcome := range []string{"damage", "freed"} {
		j, e := ReadProjectileMisses(strings.NewReader(missLog(outcome)), nil)
		if e != nil {
			t.Fatal(e)
		}
		_, s, o := rewardFixture()
		first := s
		first.Native = &NativeStep{Spawncount: 42, BeginFrame: 9, EndFrame: 10, Sequence: 22, Actor: 1}
		j.Enrich(&first, &o)
		j.Enrich(&s, &o)
		if len(o.ProjectileMisses.Misses) != 0 {
			t.Fatal(outcome, "penalized")
		}
	}
	for _, which := range []string{"unseen_launch", "launch_after_first_life", "end_after_first_life", "other_actor", "nonexclusive"} {
		j, e := ReadProjectileMisses(strings.NewReader(missLog("sky")), nil)
		if e != nil {
			t.Fatal(e)
		}
		_, s, o := rewardFixture()
		first := s
		first.Native = &NativeStep{Spawncount: 42, BeginFrame: 9, EndFrame: 10, Sequence: 22, Actor: 1}
		if which == "launch_after_first_life" {
			first.Observation.Identity.Life = 2
		}
		if which != "unseen_launch" {
			j.Enrich(&first, &o)
		}
		if which == "end_after_first_life" {
			s.Observation.Identity.Life = 2
		}
		if which == "other_actor" {
			s.Observation.Identity.Actor = 2
		}
		if which == "nonexclusive" {
			s.Execution.WindowExclusive = false
		}
		j.Enrich(&s, &o)
		if len(o.ProjectileMisses.Misses) != 0 {
			t.Fatal(which, "penalized")
		}
	}
}

func TestProjectileMissRejectsContradictoryAndMalformed(t *testing.T) {
	d := DamageEvent{Spawncount: 42, Shot: 1, Attacker: 1, Inflictor: 350, Mod: 1}
	if _, e := ReadProjectileMisses(strings.NewReader(missLog("geometry")), []DamageEvent{d}); e == nil {
		t.Fatal("damage treated as miss")
	}
	for _, bad := range []string{
		strings.Replace(missLog("geometry"), "entity=350 target", "entity=351 target", 1),
		strings.Replace(missLog("geometry"), "outcome=geometry", "outcome=unknown", 1),
		strings.Replace(missLog("geometry"), "shot=1 entity=350 target", "shot=2 entity=350 target", 1),
		strings.Replace(missLog("geometry"), "g_test_projectile ready version=1\n", "", 1),
		strings.Replace(missLog("geometry"), "server_frame=10 g_test_projectile", "server_frame=15 g_test_projectile", 1),
	} {
		if _, e := ReadProjectileMisses(strings.NewReader(bad), nil); e == nil {
			t.Fatal("invalid telemetry accepted")
		}
	}
}

func TestMissRewardV5CostAndLegacyIsolation(t *testing.T) {
	c, s, o := maneuverFixture()
	legacy := c.Evaluate(&s, o)
	if !legacy.Available {
		t.Fatal(legacy)
	}
	c.Version = MissRewardVersion
	c.BlasterMiss = -.02
	if r := c.Evaluate(&s, o); r.Available {
		t.Fatal("missing telemetry accepted")
	}
	o.ProjectileMisses = &ProjectileMissOutcome{Version: "native_projectile_miss_v1", Misses: []ProjectileMiss{}}
	r := c.Evaluate(&s, o)
	if !r.Available || *r.Score != *legacy.Score {
		t.Fatal("no-shot cost", r)
	}
	o.ProjectileMisses.Misses = []ProjectileMiss{{Shot: 1, Entity: 350, Outcome: "geometry", Launch: *s.Native, End: *s.Native}}
	r = c.Evaluate(&s, o)
	if !r.Available || math.Abs(*r.Score-*legacy.Score+.02) > 1e-12 {
		t.Fatal("cost", r)
	}
	o.ProjectileMisses.Misses = append(o.ProjectileMisses.Misses, o.ProjectileMisses.Misses[0])
	if c.Evaluate(&s, o).Available {
		t.Fatal("duplicate evidence")
	}
	c.Version = ManeuverRewardVersion
	if c.Validate() == nil {
		t.Fatal("legacy objective changed")
	}
}
