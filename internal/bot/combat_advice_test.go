package bot

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func TestCombatAdviceCannotRunInOrdinaryCoop(t *testing.T) {
	err := Run(context.Background(), Config{TestCombatAdvice: true, CombatMode: "learned"})
	if err == nil || !strings.Contains(err.Error(), "combat advice requires") {
		t.Fatalf("isolation gate missing: %v", err)
	}
}

func TestCombatAdviceActualBurstHistory(t *testing.T) {
	h := combatAdviceHistory{}
	o := policy.Observation{Identity: policy.Identity{Map: "base1", Life: 1}, Health: 100, Weapon: "Machinegun"}
	for frame := 10; frame <= 15; frame++ {
		o.Identity.Frame = frame
		if frame > 10 {
			q := Planner{machinegunBurst: h.burst}
			cmd := q.limitMachinegunBurst(quake.Snapshot{Map: "base1", Frame: frame, Health: 100, Weapon: "Machinegun"}, quake.UserCmd{Buttons: 1})
			if frame >= 13 && cmd.Buttons&1 != 0 {
				t.Fatal("expert failed to request a pause")
			}
		}
		h.observe(o, quake.UserCmd{Buttons: 1}) // learner ignores the recommendation
		if h.burst.start != 10 {
			t.Fatal("hypothetical pause changed actual history")
		}
	}
	o.Identity.Frame = 16
	h.observe(o, quake.UserCmd{})
	if h.burst.start != 10 {
		t.Fatal("one release frame cleared burst")
	}
	o.Identity.Frame = 17
	h.observe(o, quake.UserCmd{})
	if h.burst.start != 0 {
		t.Fatal("two actual releases did not clear burst")
	}
	o.Identity.Frame = 18
	h.observe(o, quake.UserCmd{Buttons: 1})
	o.Identity.Frame = 20
	h.observe(o, quake.UserCmd{Buttons: 1})
	if h.burst.start != 20 {
		t.Fatal("frame gap did not reset history")
	}
	o.Identity.Frame++
	o.Identity.Life++
	h.observe(o, quake.UserCmd{Buttons: 1})
	if h.burst.start != 21 {
		t.Fatal("new life did not reset history")
	}
	o.Identity.Frame++
	o.Weapon = "Blaster"
	h.observe(o, quake.UserCmd{Buttons: 1})
	if h.burst != (machinegunBurst{}) {
		t.Fatal("weapon change retained burst")
	}
}

func TestCombatAdviceNativeGeometryAndPurity(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	clear := true
	s := quake.Snapshot{Map: "base1", Frame: 50, Health: 100, Weapon: "Machinegun", Ammo: 100,
		Self: quake.Vec3{32, -224, 24}, OnGround: true,
		Enemies: []quake.Object{{ID: 3, Class: "monster_parasite", Origin: quake.Vec3{200, -224, 24}, Solid: 7266, ClearShot: &clear}}}
	p := Planner{Campaign: true, testCombatOnly: true, TestDisableProbe: true, TestDisableHandGrenade: true,
		World:       World{Map: "base1", Geometry: &g, GeometryStatus: "ready", Goal: "combat_fixture", Updated: now, Snapshot: s},
		enemyMotion: map[int]enemyMotion{3: {origin: s.Enemies[0].Origin, frame: 50}}}
	o := policy.Observe(s, policy.Identity{Map: "base1", Frame: 50, Life: 1}, quake.UserCmd{})
	track := 1
	for i := range o.Enemies {
		o.Enemies[i].Track = &track
	}
	before, _ := json.Marshal(p.World)
	motion := p.enemyMotion[3]
	a := p.reactiveCombatAdvice(o, quake.UserCmd{}, now, combatAdviceHistory{})
	if !a.Available || a.Action.TargetEntity != 3 || !a.Action.Attack {
		t.Fatalf("no usable expert attack: %+v", a)
	}
	if s.Teammate != nil || a.Command.Forward == 0 && a.Command.Side == 0 || a.Decision.MoveSource != "combat_retreat" {
		t.Fatalf("campaign expert must retreat from close parasite without a teammate: %+v", a)
	}
	after, _ := json.Marshal(p.World)
	if !bytes.Equal(before, after) || p.enemyMotion[3] != motion || p.machinegunBurst != (machinegunBurst{}) {
		t.Fatal("query mutated live world or planner history")
	}
	// Fresh state only: this checks the reactive recommendation, not equivalence
	// with a long-running classic planner's route/cover/burst state.
	q := Planner{Campaign: true, testCombatOnly: true, TestDisableProbe: true, TestDisableHandGrenade: true, World: p.World}
	want := q.commandAt(quake.UserCmd{}, now)
	want.Msec = 100
	if a.Command != want {
		t.Fatalf("fresh rules mismatch: %+v != %+v", a.Command, want)
	}
	o.Enemies[0].Solid = nil
	if a := p.reactiveCombatAdvice(o, quake.UserCmd{}, now, combatAdviceHistory{}); a.Available {
		t.Fatal("accepted target unavailable to policy")
	}
	p.World.Snapshot.Enemies = nil
	o.Enemies = nil
	a = p.reactiveCombatAdvice(o, quake.UserCmd{Yaw: 1234, Pitch: 999}, now, combatAdviceHistory{})
	if !a.Available || a.Command.Yaw != 1234 || a.Command.Pitch != 0 || a.Action.Attack || a.Action.TargetEntity != 0 {
		t.Fatalf("no-target hold/horizon recommendation incorrect: %+v", a)
	}
	for _, name := range []string{"equipment", "stale", "setup", "geometry", "frame"} {
		t.Run(name, func(t *testing.T) {
			cp, co := p, o
			switch name {
			case "equipment":
				co.Weapon = "Railgun"
			case "stale":
				co.AgeMS = 301
			case "setup":
				cp.testSetupHold = true
			case "geometry":
				cp.World.Geometry = nil
			case "frame":
				co.Identity.Frame++
			}
			if a := cp.reactiveCombatAdvice(co, quake.UserCmd{}, now, combatAdviceHistory{}); a.Available {
				t.Fatal("invalid advice accepted")
			}
		})
	}
}
