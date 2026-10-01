package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBunk1DeployedBridgeLink(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "bunk1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/bunk1.aas")
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range g.Entities {
		if e.Model == 128 || e.Model == 129 {
			b, _ := g.Model(e.Model)
			t.Log("door", e, b)
		}
	}
	from := quake.Vec3{-962.25, -64, 30.125}
	goal := quake.Vec3{-471.75, 486.375, 24.125}
	movers := []quake.Mover{{Model: 99, Origin: quake.Vec3{0, -250, 0}}, {Model: 100, Origin: quake.Vec3{0, 0, -202}}, {Model: 128, Origin: quake.Vec3{58, 0, 0}}, {Model: 129, Origin: quake.Vec3{-58, 0, 0}}}
	snap := quake.Snapshot{Map: "bunk1", Frame: 4700, Self: from, Teammate: &goal, Health: 87, OnGround: true, Movers: movers}
	p := &Planner{Nav: n, goalPoint: goal, World: World{Map: "bunk1", Geometry: &g, Goal: "follow_teammate", Snapshot: snap}, doorPrevious: snap}
	p.doorPrevious.Frame--
	r, task, ok := p.deployedBridgeRoute()
	if !ok {
		a, z := quake.Vec3{-512, 166, 24.125}, quake.Vec3{-512, 510, 24.125}
		t.Log("static", g.PlayerMoveClear(a, z), "corridor", p.bridgeLinkCorridor(a, z, movers[0]))
		for y := 166.; y <= 510; y += 4 {
			v := quake.Vec3{-512, y, 24.125}
			_, ok := g.GroundDrop(v, 18)
			_, mk := g.MoverFooting(movers[0], v, 18)
			if !ok && !mk {
				t.Log("unsupported", y)
			}
		}
		for _, m := range movers {
			t.Log("moverclear", m.Model, g.MoverHullClear(m, a, z))
		}
		model, reason := g.DoorMoveBlockStep(movers, a, 0, 344, 344)
		t.Log("door", model, reason)
		for _, v := range []quake.Vec3{a, z, {-512, 338, 24.125}} {
			drop, ok := g.GroundDrop(v, 36)
			foot, mok := g.MoverFooting(movers[0], v, 18)
			t.Log(v, drop, ok, foot, mok)
		}
		t.Fatal("no bridge link")
	}
	t.Log("bridge link", task, "cost", routeLength(from, r, goal))
	for _, v := range []quake.Vec3{{-878, -64, 24.125}, {-800, -64, 24.125}, {-524.875, 166, 24.125}} {
		drop, ok := g.GroundDrop(v, 4)
		t.Log("setup", v, g.PlayerMoveClear(v, v), drop, ok)
	}
	for _, wp := range r {
		if wp.Kind == 11 {
			t.Fatal("shortcut still uses elevator")
		}
	}
	original, ok := n.Route(from, goal)
	if !ok || routeLength(from, r, goal) >= routeLength(from, original, goal) {
		t.Fatal("shortcut does not improve static route")
	}
	for _, mode := range []string{"closed", "hidden", "moving", "unknown_motion", "dead", "airborne"} {
		t.Run(mode, func(t *testing.T) {
			q := *p
			q.World = p.World
			q.World.Snapshot = snap
			q.World.Snapshot.Movers = append([]quake.Mover(nil), movers...)
			q.doorPrevious = p.doorPrevious
			switch mode {
			case "closed":
				q.World.Snapshot.Movers[0].Origin = quake.Vec3{}
				q.doorPrevious.Movers = q.World.Snapshot.Movers
			case "hidden":
				q.World.Snapshot.Movers = nil
			case "moving":
				q.doorPrevious.Movers = nil
			case "unknown_motion":
				q.doorPrevious.Frame = 0
			case "dead":
				q.World.Snapshot.Health = 0
			case "airborne":
				q.World.Snapshot.OnGround = false
			}
			if _, _, ok := q.deployedBridgeRoute(); ok {
				t.Fatal("unsafe bridge accepted")
			}
		})
	}
	p.World.Snapshot.Frame--
	p.route = original
	p.routeKnown = true
	p.target = goal
	p.elevator = &elevatorRide{model: 100, stage: "ride"}
	p.update(snap, "")
	if p.elevator != nil || p.bridgeLink == nil {
		t.Fatal("stationary bottom lift was not superseded by bridge")
	}
	p.World.Snapshot.Self = quake.Vec3{-512, 330, 16.125}
	p.bridgeLink.crossing = true
	cmd, active := p.bridgeLinkCommand(quake.UserCmd{})
	if !active || cmd.Forward == 0 && cmd.Side == 0 || cmd.Up != 0 {
		t.Fatalf("cannot cross and step onto bank: %v %+v", active, cmd)
	}
}
