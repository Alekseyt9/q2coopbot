package bot

import (
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
	"time"
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

func TestCampaignBase1HatchAndContact(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Map: "base1", Frame: 100, Self: quake.Vec3{-1718.5, 1533.875, 120.125}, Health: 100, OnGround: true, Gravity: 800, Movers: []quake.Mover{{Model: 31}, {Model: 34}, {Model: 32}, {Model: 33}, {Model: 19}}}
	p := &Planner{Campaign: true, CampaignNextMap: "base2", Nav: n, GameClock: true, World: World{Map: "base1", Geometry: &g, GeometryStatus: "ready", Snapshot: s}}
	p.update(s, root)
	if len(p.World.Route) != 2 || p.World.Route[0].Kind != 7 || p.goalPoint[2] > -55 {
		t.Fatalf("lost descent pair or wrong floor: %v %v", p.World.Route, p.goalPoint)
	}
	p.doorPrevious = s
	p.doorPrevious.Frame--
	task := p.selectButtonTask(s)
	if task == nil || !task.campaign || task.doorModel != 31 || task.buttonModel != 34 {
		t.Fatal("linked hatch button not selected", task)
	}
	p.button = task
	p.validateButtonOwner(s)
	if p.button == nil {
		t.Fatal("campaign subtask requires nonexistent teammate")
	}
	s.Self = quake.Vec3{-1821.875, 1536, 120.125}
	p.World.Snapshot = s
	p.World.Updated = time.Now()
	p.button.phase = "touch"
	cmd := p.command(quake.UserCmd{})
	if cmd.Forward == 0 && cmd.Side == 0 {
		t.Fatal("stopped before contact", p.World.Command)
	}
	if p.campaignButtonStep(s, -1, 0, 8) {
		t.Fatal("contact permission extends behind button")
	}
	p.doorPrevious.Frame = 98
	if p.campaignButtonStep(s, -1, 0, 1) {
		t.Fatal("unconfirmed stationary hatch accepted")
	}
}

func TestCampaignBase1ExitStairs(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Map: "base1", Frame: 100, Self: quake.Vec3{-1619.625, 1872.125, -23.875}, Health: 104, OnGround: true, Gravity: 800}
	p := &Planner{Campaign: true, CampaignNextMap: "base2", Nav: n, GameClock: true, World: World{Map: "base1", Geometry: &g, GeometryStatus: "ready", Snapshot: s}}
	p.update(s, root)
	wp := p.World.Route[0].Position
	if g.GroundMoveHazardStep(n, s.Self, wp[0]-s.Self[0], wp[1]-s.Self[1], 8) != "static_hull_blocked" {
		t.Fatal("fixture no longer reproduces blocked stair corner")
	}
	dx, dy, ok := p.regroupCornerStep(s, wp)
	if !ok || g.GroundMoveHazardStep(n, s.Self, dx, dy, 16) != "" || g.DoorMoveHazard(s.Movers, s.Self, dx, dy) != "" {
		t.Fatal("no checked corner bypass", dx, dy, ok)
	}
	p.World.Updated = time.Now()
	cmd := p.command(quake.UserCmd{})
	if cmd.Forward == 0 && cmd.Side == 0 || p.World.Command.MoveSource != "route_corner_bypass" {
		t.Fatal("blocked exit stairs", p.World.Command, p.World.Route[:2])
	}
}
