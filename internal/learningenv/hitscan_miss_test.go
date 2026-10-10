package learningenv

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func hitscanLog() string {
	return "sv_test_step version=1 phase=begin spawncount=42 frame=10 seq=23 actor=1\n" +
		"sv_test_hitscan spawncount=42 server_frame=10 g_test_hitscan version=1 event=fire map=base1 frame=12 shot=1 actor=1 mod=4 start=0,0,0 aim=1,0,0 view=0,0,0 recoil=0,0,0 velocity=0,0,0 spread=0,0 nominal=300,500 water=0 muzzle_blocked=0 fraction=0.5 impact=100,0,0 sky=0 target=0 target_class=worldspawn damageable=0 health_before=0 burst=0 gunframe=4 ammo_before=50 damage=8\n" +
		"sv_test_step version=1 phase=end spawncount=42 frame=11 seq=23 actor=1\n"
}

func TestHitscanMissExactlyOnceAndMasks(t *testing.T) {
	for _, mask := range []string{"none", "later_life", "dead", "recovery", "nonexclusive", "wrong_sequence", "wrong_actor", "wrong_generation"} {
		t.Run(mask, func(t *testing.T) {
			j, err := ReadHitscanMisses(strings.NewReader(hitscanLog()), nil)
			if err != nil {
				t.Fatal(err)
			}
			_, s, o := rewardFixture()
			switch mask {
			case "later_life":
				s.Observation.Identity.Life = 2
			case "dead":
				s.Observation.Health = 0
			case "recovery":
				s.Execution.RecoveryCommands = 1
			case "nonexclusive":
				s.Execution.WindowExclusive = false
			case "wrong_sequence":
				s.Native.Sequence++
			case "wrong_actor":
				s.Native.Actor++
			case "wrong_generation":
				s.Native.Spawncount++
			}
			if err := j.Enrich(&s, &o); err != nil {
				t.Fatal(err)
			}
			if mask == "none" {
				if len(o.HitscanMisses.Misses) != 1 {
					t.Fatal(o)
				}
				if err := j.Enrich(&s, &o); err == nil {
					t.Fatal("duplicate credit")
				}
			} else if len(o.HitscanMisses.Misses) != 0 {
				t.Fatal("masked miss charged", o)
			}
		})
	}
}

func TestHitscanMissRejectsMalformedAndContradictory(t *testing.T) {
	for _, pair := range [][2]string{
		{"mod=4", "mod=3"}, {"shot=1", "shot=0"}, {"fraction=0.5", "fraction=NaN"},
		{"aim=1,0,0", "aim=0,1,0"}, {"nominal=300,500", "nominal=-1,500"},
		{"spread=0,0", "spread=301,0"}, {"ammo_before=50", "ammo_before=0"},
		{"server_frame=10", "server_frame=12"}, {"actor=1 mod=4", "actor=1 actor=1 mod=4"},
		{"impact=100,0,0 ", ""}, {"phase=end spawncount=42", "phase=end spawncount=43"},
	} {
		if _, err := ReadHitscanMisses(strings.NewReader(strings.Replace(hitscanLog(), pair[0], pair[1], 1)), nil); err == nil {
			t.Fatal("invalid accepted", pair)
		}
	}
	line := strings.Split(hitscanLog(), "\n")[1] + "\n"
	duplicate := strings.Replace(hitscanLog(), line, line+line, 1)
	if _, err := ReadHitscanMisses(strings.NewReader(duplicate), nil); err == nil {
		t.Fatal("duplicate shot accepted")
	}
	d := DamageEvent{Spawncount: 42, Frame: 10, GameFrame: 12, Map: "base1", Mod: 4, Attacker: 1, Inflictor: 1, Target: 0, HealthBefore: 0, HealthAfter: -8, Take: 8}
	damageLine := "sv_test_damage spawncount=42 server_frame=10 g_test_damage version=1 map=base1 frame=12 attacker=1 target=0 inflictor=1 mod=4 health_before=0 health_after=-8 take=8 armor=0 power=0 protection=0 target_class=worldspawn attacker_class=player\n"
	bad := strings.Replace(hitscanLog(), line, line+damageLine, 1)
	if _, err := ReadHitscanMisses(strings.NewReader(bad), []DamageEvent{d}); err == nil {
		t.Fatal("geometry with damage accepted")
	}
}

func TestHitscanMissLivingContactUnknownAndCorpse(t *testing.T) {
	for _, health := range []int{100, 0, -10} {
		text := strings.Replace(hitscanLog(), "damageable=0 health_before=0", fmt.Sprintf("damageable=2 health_before=%d", health), 1)
		j, err := ReadHitscanMisses(strings.NewReader(text), nil)
		if err != nil {
			t.Fatal(err)
		}
		_, s, o := rewardFixture()
		if err := j.Enrich(&s, &o); err != nil {
			t.Fatal(err)
		}
		want := 0
		if health <= 0 {
			want = 1
		}
		if len(o.HitscanMisses.Misses) != want {
			t.Fatal("ambiguous living contact/corpse", health, o)
		}
	}
}

func TestNativeWasteRewardV9CostsAndV8Isolation(t *testing.T) {
	c, s, o := selectedAimFixture()
	before := c.Evaluate(&s, o)
	if !before.Available {
		t.Fatal(before)
	}
	c.Version, c.BlasterMiss, c.MachinegunMiss = NativeWasteRewardVersion, -.02, -.02
	if c.Evaluate(&s, o).Available {
		t.Fatal("missing native evidence accepted")
	}
	o.ProjectileMisses = &ProjectileMissOutcome{Version: "native_projectile_miss_v1", Misses: []ProjectileMiss{}}
	o.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1", Misses: []HitscanMiss{}}
	empty := c.Evaluate(&s, o)
	if !empty.Available || *empty.Score != *before.Score || empty.AimReference == nil || empty.AimReference.Entity != before.AimReference.Entity {
		t.Fatal("v8 aim/objective changed without shots", empty, before)
	}
	o.ProjectileMisses.Misses = []ProjectileMiss{{Shot: 1, Entity: 350, Outcome: "sky", Launch: *s.Native, End: *s.Native}}
	o.HitscanMisses.Misses = []HitscanMiss{{Shot: 1, Outcome: "geometry", Window: *s.Native}}
	after := c.Evaluate(&s, o)
	if !after.Available || math.Abs(*after.Score-*before.Score+.04) > 1e-12 {
		t.Fatal("wrong shot cost", after)
	}
	o.HitscanMisses.Misses = append(o.HitscanMisses.Misses, o.HitscanMisses.Misses[0])
	if c.Evaluate(&s, o).Available {
		t.Fatal("duplicate miss accepted")
	}
	c.Version = SelectedAimRewardVersion
	if c.Validate() == nil {
		t.Fatal("v8 accepts new objective coefficients")
	}
}

func TestNativeWasteRewardRetainsDeathAndGoalTerminal(t *testing.T) {
	for _, death := range []bool{false, true} {
		c, s, o := selectedAimFixture()
		s.Terminal = true
		if death {
			s.Next.Health = 0
			o.Deaths = 1
		} else {
			s.Reason = "combat_goal_complete"
		}
		before := c.Evaluate(&s, o)
		if !before.Available {
			t.Fatal(before)
		}
		c.Version, c.BlasterMiss, c.MachinegunMiss = NativeWasteRewardVersion, -.02, -.02
		o.ProjectileMisses = &ProjectileMissOutcome{Version: "native_projectile_miss_v1", Misses: []ProjectileMiss{}}
		o.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1", Misses: []HitscanMiss{{Shot: 1, Outcome: "geometry", Window: *s.Native}}}
		after := c.Evaluate(&s, o)
		if !after.Available || after.Components["death"] != before.Components["death"] || after.Components["aim_potential"] != 0 || math.Abs(*after.Score-*before.Score+.02) > 1e-12 {
			t.Fatal("terminal reward lost", death, after)
		}
	}
}
