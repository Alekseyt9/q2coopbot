package bot

import (
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestRespawnWaitPulseReleaseAndReset(t *testing.T) {
	p := &Planner{World: World{Map: "test", GeometryStatus: "ready", Updated: time.Now()}}
	for _, tc := range []struct {
		frame  int
		health int16
		press  bool
	}{{100, -12, false}, {109, -12, false}, {110, -12, true}, {111, -12, false}, {115, -12, true}, {116, 100, false}} {
		p.update(quake.Snapshot{Map: "test", Frame: tc.frame, Health: tc.health}, "")
		cmd := p.commandAt(quake.UserCmd{}, p.World.Updated)
		if (cmd.Buttons == 1) != tc.press || cmd.Forward != 0 || cmd.Side != 0 || cmd.Up != 0 {
			t.Fatalf("frame %d: %+v", tc.frame, cmd)
		}
	}
	p.update(quake.Snapshot{Map: "test", Frame: 117, Health: 100}, "")
	p.World.Snapshot.Health = -10
	p.World.Snapshot.Frame = 200
	// A new death must get a new delay even if healthy snapshots were processed.
	if p.deathFrame != 0 {
		t.Fatal("healthy update retained death clock")
	}
	if cmd := p.commandAt(quake.UserCmd{}, p.World.Updated); cmd.Buttons != 0 {
		t.Fatal("respawn fired before delay")
	}
}

func TestUnreachableHealthBudgetAndCooldown(t *testing.T) {
	mate := quake.Vec3{200, 0, 24}
	item := quake.Vec3{60, 0, 15}
	s := quake.Snapshot{Map: "test", Frame: 40, Health: 36, Teammate: &mate, Self: quake.Vec3{0, 0, 24}}
	p := &Planner{World: World{Goal: "recover_health"}}
	target := healthStand(item)
	if got := p.budgetHealthGoal(s, target); got != target {
		t.Fatal("cancelled before budget")
	}
	s.Frame = 65
	if got := p.budgetHealthGoal(s, target); got != mate || p.World.Goal != "follow_teammate" || p.healthAllowed(item, 66) {
		t.Fatal("unreachable pickup retained")
	}
	if !p.healthAllowed(item, 215) {
		t.Fatal("cooldown never expires")
	}
	p.setMap("new", "")
	if !p.healthAllowed(item, 1) {
		t.Fatal("pickup cooldown leaked across maps")
	}
}

func TestHealthChoiceDoesNotDependOnEntityIterationOrder(t *testing.T) {
	mate := quake.Vec3{200, 0, 24}
	a := quake.Object{ID: 2, Class: "item_health", Origin: quake.Vec3{60, 0, 15}}
	b := quake.Object{ID: 3, Class: "item_health", Origin: quake.Vec3{90, 0, 15}}
	p := &Planner{World: World{Map: "test"}}
	for frame := 40; frame < 65; frame++ {
		items := []quake.Object{a, b}
		if frame%2 == 0 {
			items = []quake.Object{b, a}
		}
		p.update(quake.Snapshot{Map: "test", Frame: frame, Health: 36, Teammate: &mate, Pickups: items}, "")
		if p.goalPoint != healthStand(a.Origin) || p.healthAt != 40 {
			t.Fatalf("unstable target at %d: %v, budget %d", frame, p.goalPoint, p.healthAt)
		}
	}
}
