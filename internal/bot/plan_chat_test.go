package bot

import (
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func TestPlanChatDebounceCooldownAndGeneration(t *testing.T) {
	p := &Planner{World: World{Navigation: "ready", Goal: "recover_health", Snapshot: quake.Snapshot{Map: "base1", Frame: 10, Health: 30}}}
	now := time.Unix(1000, 0)
	c := planChat{}
	if c.command(p, now) != "" {
		t.Fatal("no debounce")
	}
	p.World.Snapshot.Frame = 20
	if got := c.command(p, now); got != "say \"Idu za aptechkoy.\"" {
		t.Fatal(got)
	}
	p.World.Goal = "collect_item"
	p.World.Snapshot.Frame = 30
	c.command(p, now)
	p.World.Snapshot.Frame = 40
	if c.command(p, now.Add(19*time.Second)) != "" {
		t.Fatal("chat spam")
	}
	if c.command(p, now.Add(20*time.Second)) == "" {
		t.Fatal("changed plan lost")
	}
	p.World.Goal = "recover_health"
	p.World.Snapshot.Frame = 50
	c.command(p, now)
	p.World.Snapshot.Frame = 60
	if c.command(p, now.Add(40*time.Second)) != "" {
		t.Fatal("repeated plan spam")
	}
	p.World.Snapshot.Map = "base2"
	p.World.Snapshot.Frame = 1
	if c.command(p, now.Add(121*time.Second)) != "" {
		t.Fatal("map generation skips debounce")
	}
	p.World.Snapshot.Frame = 11
	if c.command(p, now.Add(121*time.Second)) == "" {
		t.Fatal("frame reset broke chat")
	}
	p.World.Snapshot.Health = 0
	if c.command(p, now.Add(300*time.Second)) != "" {
		t.Fatal("dead bot announces a plan")
	}
}

func TestExitPreparationOnlyNearExitWithHealthNeed(t *testing.T) {
	p := &Planner{Campaign: true, World: World{Campaign: &CampaignDecision{Exit: &quake.MapExit{Center: quake.Vec3{100, 0, 0}}}}}
	s := quake.Snapshot{Health: 60}
	if !p.preparingForExit(s) {
		t.Fatal("missed exit preparation")
	}
	for _, s := range []quake.Snapshot{{Health: 75}, {Health: 60, Self: quake.Vec3{1000, 0, 0}}, {Health: 60, Teammate: &quake.Vec3{}}} {
		if p.preparingForExit(s) {
			t.Fatal("unexpected detour", s)
		}
	}
}

func TestCampaignPreparationUsesRememberedKitAboveCriticalHealth(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Map: "base1", Frame: 233, Health: 60, OnGround: true, Self: quake.Vec3{-1615.875, 1865.375, -7.875}}
	kit := quake.Object{ID: 227, Class: "item_health", Origin: quake.Vec3{-1176, 1520, -32.875}, HealthAmount: 25}
	p := &Planner{Campaign: true, CampaignNextMap: "base2", Nav: n, World: World{Map: "base1", Geometry: &g, GeometryStatus: "ready"}, resources: map[int]*ResourceMemory{227: {Item: kit, LastSeen: 218, State: "unknown"}}}
	p.update(s, root)
	if p.World.Goal != "recover_health" || p.goalPoint != healthStand(kit.Origin) {
		t.Fatal("noncritical exit preparation did not use remembered kit", p.World.Goal, p.goalPoint)
	}
	key, text := planChatText(p)
	if key != "prepare" || text != "Sobirayu resursy pered perehodom." {
		t.Fatal(key, text)
	}
}
