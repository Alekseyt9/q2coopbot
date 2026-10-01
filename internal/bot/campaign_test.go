package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestCampaignExitSelectionAndActualMapCompletion(t *testing.T) {
	m := &quake.MapInfo{Models: []quake.BSPModel{{}, {Min: quake.Vec3{100, 0, -24}, Max: quake.Vec3{160, 64, 64}}}, Entities: []quake.MapEntity{{Class: "trigger_multiple", Model: 1, Target: "exit"}, {Class: "target_changelevel", TargetName: "exit", Map: "base2$base1"}}}
	p := &Planner{Campaign: true, World: World{Geometry: m}}
	s := quake.Snapshot{Map: "base1", Self: quake.Vec3{80, 32, 0}, Health: 100}
	goal, ok := p.campaignGoal(s)
	if !ok || goal != (quake.Vec3{130, 32, 0}) || p.World.Campaign.Exit.Destination != "base2$base1" {
		t.Fatal(goal, p.World.Campaign)
	}
	s.Self = goal
	if _, ok = p.campaignGoal(s); !ok || p.World.Campaign.State == "level_completed" {
		t.Fatal("touching coordinates is not completion")
	}
	s.Map = "base2"
	if _, ok = p.campaignGoal(s); ok || p.World.Campaign.State != "level_completed" {
		t.Fatal(p.World.Campaign)
	}
	s.Map = "base3"
	if _, ok = p.campaignGoal(s); ok || p.World.Campaign.State != "unexpected_map_change" {
		t.Fatal(p.World.Campaign)
	}
	m.Entities = append(m.Entities, quake.MapEntity{Class: "trigger_changelevel", Map: "base3", Model: 1})
	p = &Planner{World: World{Geometry: m}}
	s.Map = "base1"
	if _, ok = p.campaignGoal(s); ok || p.World.Campaign.State != "ambiguous_exits" {
		t.Fatal("ambiguous exit accepted")
	}
	p.CampaignNextMap = "base3"
	if _, ok = p.campaignGoal(s); !ok || p.World.Campaign.Exit.Destination != "base3" {
		t.Fatal("explicit destination ignored")
	}
}
