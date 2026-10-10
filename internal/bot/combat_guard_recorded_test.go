package bot

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"q2coopbot/internal/quake"
)

// Reproduce a native command without evaluating a neural network.
func TestCombatGuardRecordedDoor(t *testing.T) {
	path := "testdata/combat-door-base1-frame129.json"
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires BSP assets")
	}
	var r struct {
		Map      string
		Self     quake.Vec3
		Velocity quake.Vec3 `json:"self_velocity"`
		OnGround bool       `json:"on_ground"`
		Ducked   bool
		Delta    [3]int16 `json:"delta_angles"`
		Health   int16
		Weapon   string
		Movers   []quake.Mover
		Combat   struct {
			Command quake.UserCmd `json:"proposed_command"`
		} `json:"combat_policy"`
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	g, err := quake.LoadMap(root, r.Map)
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Self: r.Self, SelfVelocity: r.Velocity, OnGround: r.OnGround, Ducked: r.Ducked, DeltaAngles: r.Delta, Health: r.Health, Weapon: r.Weapon, Movers: r.Movers}
	cmd := r.Combat.Command
	cmd.Msec = 100
	probe := cmd
	probe.Up = 0
	p := predictGroundStep(s, probe)
	if p == nil {
		t.Fatal("ground prediction unavailable")
	}
	_, actual := g.DoorMoveBlockStep(s.Movers, s.Self, p.Displacement[0], p.Displacement[1], math.Hypot(p.Displacement[0], p.Displacement[1]))
	t.Logf("self=%v cmd=%+v prediction=%+v old=%s bounded=%s", s.Self, cmd, p, g.DoorMoveHazard(s.Movers, s.Self, p.Displacement[0], p.Displacement[1]), actual)
	s.Ducked = true
	p = predictGroundStep(s, probe)
	_, actual = g.DoorMoveBlockStep(s.Movers, s.Self, p.Displacement[0], p.Displacement[1], math.Hypot(p.Displacement[0], p.Displacement[1]))
	t.Logf("crouch prediction=%+v bounded=%s static=%s", p, actual, g.GroundMoveHazardStep(nil, s.Self, p.Displacement[0], p.Displacement[1], math.Hypot(p.Displacement[0], p.Displacement[1])))
	for _, axis := range []string{"forward", "side"} {
		part := probe
		if axis == "forward" {
			part.Side = 0
		} else {
			part.Forward = 0
		}
		p = predictGroundStep(s, part)
		_, actual = g.DoorMoveBlockStep(s.Movers, s.Self, p.Displacement[0], p.Displacement[1], math.Hypot(p.Displacement[0], p.Displacement[1]))
		t.Logf("component %s displacement=%v door=%s static=%s", axis, p.Displacement, actual, g.GroundMoveHazardStep(nil, s.Self, p.Displacement[0], p.Displacement[1], math.Hypot(p.Displacement[0], p.Displacement[1])))
	}
	s.Ducked = r.Ducked
	planner := &Planner{World: World{Geometry: &g}}
	got, changes := planner.guardDirectCombat(s, cmd)
	if got.Forward != 0 || got.Side != cmd.Side || got.Up != cmd.Up || got.Pitch != cmd.Pitch || got.Yaw != cmd.Yaw || got.Buttons != cmd.Buttons || len(changes) != 1 || changes[0].Reason != "dynamic_door_component_clipped" {
		t.Fatalf("requested safe crouch strafe was not preserved: %+v %+v", got, changes)
	}
	if _, ok := planner.checkedCombatMovementComponent(s, cmd, "dynamic_door_unobserved"); ok {
		t.Fatal("unobserved door accepted")
	}
	jump := cmd
	jump.Up = 400
	if _, ok := planner.checkedCombatMovementComponent(s, jump, "dynamic_door_blocked"); ok {
		t.Fatal("jump trajectory accepted")
	}
	s.OnGround = false
	if _, ok := planner.checkedCombatMovementComponent(s, cmd, "dynamic_door_blocked"); ok {
		t.Fatal("airborne trajectory accepted")
	}
}

func TestCombatGuardRequestedCrouchSpeed(t *testing.T) {
	s := quake.Snapshot{Health: 100, OnGround: true}
	cmd := quake.UserCmd{Forward: 400, Up: -400, Msec: 100}
	p := predictCombatGroundStep(s, cmd)
	if p == nil || math.Abs(p.Displacement[0]-10) > 1e-9 {
		t.Fatalf("crouch command did not cap the same tick at100: %+v", p)
	}
	s.Ducked = true
	cmd.Up = 0
	p = predictCombatGroundStep(s, cmd)
	if p == nil || math.Abs(p.Displacement[0]-30) > 1e-9 {
		t.Fatalf("release understated possible standing speed: %+v", p)
	}
	s.OnGround = false
	cmd.Up = -400
	if predictCombatGroundStep(s, cmd) != nil {
		t.Fatal("airborne crouch treated as ground motion")
	}
}
