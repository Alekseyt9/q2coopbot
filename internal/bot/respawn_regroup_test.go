package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestRespawnRegroupLongRouteAndLifecycle(t *testing.T) {
	last := quake.Vec3{2000, 0, 200}
	n := &quake.Navigator{Areas: []quake.Area{{}, {Min: quake.Vec3{-10, -10, -10}, Max: quake.Vec3{10, 10, 40}}, {Min: quake.Vec3{1990, -10, 180}, Max: quake.Vec3{2010, 10, 240}}}, Edges: [][]quake.Edge{{}, {{To: 2, Start: quake.Vec3{32, 0, 24}, End: last, Kind: 2, Cost: 10}}, nil}}
	p := &Planner{Nav: n, World: World{Map: "test", GeometryStatus: "ready", Snapshot: quake.Snapshot{Map: "test", Frame: 100, Health: -5}}}
	age := 600
	s := quake.Snapshot{Map: "test", Frame: 101, Health: 100, OnGround: true, Self: quake.Vec3{0, 0, 24}, LastTeammate: &last, LastTeammateEntity: 2, TeammateAgeFrames: &age}
	p.update(s, "")
	if p.World.Goal != "regroup_after_respawn" || p.World.Navigation != "ready" || !p.hasGoal || p.World.Snapshot.Teammate != nil || p.World.SearchRoute != nil {
		t.Fatalf("no long regroup: %+v", p.World)
	}
	cmd := p.command(quake.UserCmd{})
	if cmd.Forward == 0 && cmd.Side == 0 {
		t.Fatal("regroup goal cannot move")
	}
	s.Frame = 1000
	age = 1499
	p.update(s, "")
	if p.World.Goal != "regroup_after_respawn" {
		t.Fatal("regroup expired with old sighting")
	}
	s.Self = last
	s.Frame++
	p.update(s, "")
	if p.respawnRegroup != nil || p.hasGoal {
		t.Fatal("kept walking after reaching remembered location")
	}
	for _, kind := range []string{"visible", "disconnect", "entity", "map", "rewind"} {
		t.Run(kind, func(t *testing.T) {
			q := &Planner{respawnRegroup: &respawnRegroup{entity: 2, target: last}}
			v := s
			v.Frame = 1002
			switch kind {
			case "visible":
				v.Teammate = &last
			case "disconnect":
				v.LastTeammate = nil
			case "entity":
				v.LastTeammateEntity = 3
			case "map":
				v.Map = "other"
			case "rewind":
				v.Frame = 1
			}
			q.observeRespawnRegroup(s, v)
			if q.respawnRegroup != nil {
				t.Fatal("stale regroup retained")
			}
		})
	}
}
