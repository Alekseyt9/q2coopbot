package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase3ShortTravelEffectRequiresClearCorridor(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires original base3 BSP")
	}
	g, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	p := &Planner{World: World{Geometry: &g}}
	from, to := quake.Vec3{-484, -360, -263.875}, quake.Vec3{-416, -320, -311.875}
	for _, tc := range []struct {
		name   string
		offset quake.Vec3
		want   bool
	}{
		{"closed", quake.Vec3{}, false},
		{"partial", quake.Vec3{0, 4, 0}, false},
		{"clear_below_old_limit", quake.Vec3{0, 56, 0}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			live := quake.Mover{Model: 21, Origin: tc.offset}
			s := quake.Snapshot{Movers: []quake.Mover{live}}
			if got := p.buttonDoorPassable(s, live, quake.Vec3{}, from, to); got != tc.want {
				t.Fatalf("passable=%t, want %t", got, tc.want)
			}
		})
	}
	// Observing a raised platform is not proof the old descent is still open.
	live := quake.Mover{Model: 20, Origin: quake.Vec3{0, 0, 50}}
	if p.buttonDoorPassable(quake.Snapshot{Movers: []quake.Mover{live}}, live, quake.Vec3{}, from, to) {
		t.Fatal("raised platform falsely cleared the descending corridor")
	}
}

func TestPermanentButtonEffectObservationAndReset(t *testing.T) {
	g := &quake.MapInfo{Entities: []quake.MapEntity{{Class: "func_button", Model: 34, Wait: -1}}}
	task := &buttonTask{campaign: true, action: "shoot", phase: "shoot", buttonModel: 34, doorModel: 33, started: 1}
	p := &Planner{button: task, World: World{Map: "a", Geometry: g, GeometryStatus: "ready"}}
	s := quake.Snapshot{Frame: 10, Health: 100, Movers: []quake.Mover{{Model: 34}, {Model: 33}}}
	p.applyButtonTask(s)
	if len(p.buttonEffects) != 0 {
		t.Fatal("attempt alone consumed permanent button")
	}
	s.Movers[0].Origin[1] = 4
	s.Frame = 12
	p.applyButtonTask(s)
	key := buttonEffectKey{34, 33}
	effect := p.buttonEffects[key]
	if effect == nil || effect.PressedFrame != 12 || effect.State != "awaiting_door_effect" {
		t.Fatal(effect)
	}
	p.cancelButtonTask(13)
	s.Frame = 200
	s.Movers = nil
	p.updateButtonEffects(s)
	if effect.State != "awaiting_door_effect" || p.buttonEffects[key] == nil {
		t.Fatal("PVS loss fabricated failure or reset")
	}
	s.Movers = []quake.Mover{{Model: 33}}
	p.updateButtonEffects(s)
	if effect.State != "door_effect_not_observed" {
		t.Fatal("missing effect not diagnosed", effect)
	}
	s.Health = 0
	p.validateButtonOwner(s)
	if p.buttonEffects[key] == nil {
		t.Fatal("death forgot consumed activation")
	}
	s.Frame = 210
	s.Movers[0].Origin[2] = 74
	p.routeKnown = true
	p.updateButtonEffects(s)
	if effect.State != "door_effect_observed" || p.routeKnown {
		t.Fatal("late opening did not recover", effect)
	}
	s.Movers = []quake.Mover{{Model: 34}}
	p.updateButtonEffects(s)
	if p.buttonEffects[key] != nil {
		t.Fatal("contradictory button reset ignored")
	}
	p.rememberButtonEffect(task, 220)
	p.setMap("b", "")
	if len(p.buttonEffects) != 0 {
		t.Fatal("consumption leaked across maps")
	}
}

func TestTemporaryButtonCanBeRetried(t *testing.T) {
	p := &Planner{World: World{Geometry: &quake.MapInfo{Entities: []quake.MapEntity{{Class: "func_button", Model: 2, Wait: 1}}}}}
	p.rememberButtonEffect(&buttonTask{buttonModel: 2, doorModel: 1}, 10)
	if len(p.buttonEffects) != 0 {
		t.Fatal("temporary button permanently consumed")
	}
}

func TestPermanentTouchButtonStopsAfterMovement(t *testing.T) {
	p := &Planner{button: &buttonTask{campaign: true, action: "touch", phase: "touch", buttonModel: 2, doorModel: 1, started: 1}, World: World{GeometryStatus: "ready", Geometry: &quake.MapInfo{Entities: []quake.MapEntity{{Class: "func_button", Model: 2, Wait: -1}}}}}
	s := quake.Snapshot{Frame: 10, Health: 100, Movers: []quake.Mover{{Model: 2, Origin: quake.Vec3{0, 4, 0}}, {Model: 1}}}
	p.applyButtonTask(s)
	cmd, active := p.buttonShotCommand(s, quake.UserCmd{}, quake.UserCmd{Forward: 400, Buttons: 1})
	if !active || cmd.Forward != 0 || cmd.Buttons != 0 || p.button.phase != "wait_effect" {
		t.Fatal("continued pushing consumed touch button", cmd)
	}
}
