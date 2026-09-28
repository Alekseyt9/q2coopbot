package bot

import (
	"math"
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestBase3ReturnLip(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, e := quake.LoadMap(root, "base3")
	if e != nil {
		t.Fatal(e)
	}
	n, e := quake.LoadAAS(root + "/maps/base3.aas")
	if e != nil {
		t.Fatal(e)
	}
	s := quake.Snapshot{Map: "base3", Frame: 100, Health: 100, OnGround: true, Self: quake.Vec3{1518, 1400.25, -807.875}}
	goal := quake.Vec3{-766.25, -774.5, -279.875}
	p := &Planner{Nav: n, World: World{Map: s.Map, Geometry: &g, GeometryStatus: "ready", Snapshot: s, Goal: "regroup_after_respawn"}, goalPoint: goal}
	p.World.Route, _ = n.Route(s.Self, goal)
	if !p.planGapJump() {
		t.Fatal("no safe crossing")
	}
	if p.jump.speed >= 200 || p.jump.landing[0] >= 1440 || p.jump.landing[2] >= s.Self[2] {
		t.Fatalf("wrong landing: %+v", p.jump)
	}
	route := p.World.Route
	p.World.Route = append([]quake.Waypoint{{Kind: 11}}, route...)
	p.jump = nil
	if p.planGapJump() {
		t.Fatal("must not jump across an untraversed elevator")
	}
	p.World.Route = route
	p.World.Goal = "follow_teammate"
	if !p.planGapJump() {
		t.Fatal("same crossing must work while following")
	}
}

func TestBlockedWalkingCornerDoesNotAuthorizeJump(t *testing.T) {
	p := &Planner{World: World{Snapshot: quake.Snapshot{Self: quake.Vec3{641.75, 2504.875, -231.875}}, Route: []quake.Waypoint{{Position: quake.Vec3{640, 2521, -232}, Kind: 2}, {Position: quake.Vec3{640, 2526, -231.875}, Kind: 2}}}}
	if p.blockedDropApproach() {
		t.Fatal("ordinary walking corner must not start a gap jump")
	}
	p.World.Snapshot.Self = quake.Vec3{1518, 1400.25, -807.875}
	p.World.Route = []quake.Waypoint{{Position: quake.Vec3{1441, 1371, -780}, Kind: 7, ToArea: 490}, {Position: quake.Vec3{1439, 1371, -840}, Kind: 7, ToArea: 490}}
	if !p.blockedDropApproach() {
		t.Fatal("nearby paired drop not recognized")
	}
	p.World.Route[1].ToArea++
	if p.blockedDropApproach() {
		t.Fatal("unpaired reaches accepted")
	}
}

func TestBase3ReturnRampJump(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base3.aas")
	if err != nil {
		t.Fatal(err)
	}
	self := quake.Vec3{1310.625, 1526.875, -831.25}
	goal := quake.Vec3{-148.375, -802.875, -231.875}
	route, ok := n.Route(self, goal)
	if !ok {
		t.Fatal("missing AAS route")
	}
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base3", Frame: 100, Health: 100, OnGround: true, Self: self}, Route: route}, goalPoint: goal}
	if !p.blockedRiseApproach() {
		t.Fatalf("not an upward walking reach: %+v", route[:2])
	}
	if !p.planRampJump() {
		t.Fatal("no verified ramp crossing")
	}
	if p.jump.landing[0] >= 1280 || p.jump.landing[1] < 1540 || p.jump.landing[2]-self[2] > 40 || p.jump.speed >= 200 {
		t.Fatalf("wrong ramp landing: %+v", p.jump)
	}
}

func TestBase3ReturnCornerStep(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base3.aas")
	if err != nil {
		t.Fatal(err)
	}
	self := quake.Vec3{218.25, -223.375, -493.375}
	goal := quake.Vec3{-148.375, -802.875, -231.875}
	route, ok := n.Route(self, goal)
	if !ok || len(route) < 2 {
		t.Fatal("missing AAS route")
	}
	s := quake.Snapshot{Map: "base3", Frame: 100, Health: 100, OnGround: true, Self: self}
	p := &Planner{Nav: n, World: World{Geometry: &g, Snapshot: s, Goal: "regroup_after_respawn", Route: route}, goalPoint: goal}
	dx, dy, ok := p.regroupCornerStep(s, route[1].Position)
	if !ok || dx <= 0 || dy <= 0 || g.GroundMoveHazardStep(n, self, dx, dy, 16) != "" {
		t.Fatalf("no safe sidestep: %v,%v,%v", dx, dy, ok)
	}
	p.World.Goal = "follow_teammate"
	if _, _, ok = p.regroupCornerStep(s, route[1].Position); ok {
		t.Fatal("unscoped sidestep")
	}
}

func TestBase3ReturnSecondRampPlan(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base3.aas")
	if err != nil {
		t.Fatal(err)
	}
	self := quake.Vec3{155.25, 681.375, -743.875}
	goal := quake.Vec3{-148.375, -802.875, -231.875}
	route, ok := n.Route(self, goal)
	if !ok {
		t.Fatal("missing route")
	}
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base3", Frame: 100, Health: 100, OnGround: true, Self: self, SelfVelocity: quake.Vec3{-188, -242, 0}}, Route: route}, goalPoint: goal}
	if !p.planGapJump() {
		t.Fatal("missing verified jump")
	}
	if p.jump.landing[0] > 100 || p.jump.landing[2] <= self[2] {
		t.Fatalf("wrong upward ramp target: %+v", *p.jump)
	}
	if p.jump.phase == 2 {
		dx, dy := p.jump.landing[0]-self[0], p.jump.landing[1]-self[1]
		along := (-188*dx - 242*dy) / math.Hypot(dx, dy)
		if along > p.jump.speed+20 {
			t.Fatalf("unsafe direct takeoff: %+v", *p.jump)
		}
	}
}

func TestBase3ReturnThirdRampPlan(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base3.aas")
	if err != nil {
		t.Fatal(err)
	}
	self := quake.Vec3{59.125, 624.375, -727.875}
	goal := quake.Vec3{-148.375, -802.875, -231.875}
	route, ok := n.Route(self, goal)
	if !ok {
		t.Fatal("missing route")
	}
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base3", Frame: 100, Health: 100, OnGround: true, Self: self}, Route: route}, goalPoint: goal}
	for len(p.World.Route) > 0 && quake.Horizontal(self, p.World.Route[0].Position) <= 10 && math.Abs(self[2]-p.World.Route[0].Position[2]) <= 8 {
		p.World.Route = p.World.Route[1:]
	}
	if p.World.Route[0].Position[2]-self[2] < 8 {
		t.Fatal("uphill waypoint skipped")
	}
	dx, dy, step := p.regroupCornerStep(p.World.Snapshot, p.World.Route[0].Position)
	if !step || dx <= 0 || g.GroundMoveHazardStep(n, self, dx, dy, 16) != "" {
		t.Fatalf("no safe local sidestep: %v,%v,%v", dx, dy, step)
	}
}

func TestBase3FirstReturnRampMomentum(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base3.aas")
	if err != nil {
		t.Fatal(err)
	}
	self := quake.Vec3{298.125, 725.75, -807.875}
	goal := quake.Vec3{-148.375, -802.875, -231.875}
	route, ok := n.Route(self, goal)
	if !ok {
		t.Fatal("missing route")
	}
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base3", Frame: 226, Health: 100, OnGround: true, Self: self, SelfVelocity: quake.Vec3{-229.125, -193.625, 0}}, Route: route}, goalPoint: goal}
	if !p.planRampJump() {
		t.Fatal("missing ramp jump")
	}
	if p.jump.phase == 2 {
		dx, dy := p.jump.landing[0]-self[0], p.jump.landing[1]-self[1]
		along := (-229.125*dx - 193.625*dy) / math.Hypot(dx, dy)
		if along > p.jump.speed+20 {
			t.Fatalf("launched with excessive approach momentum: %+v", *p.jump)
		}
	}
}

func TestBase3LaterReturnRampMomentum(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base3")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base3.aas")
	if err != nil {
		t.Fatal(err)
	}
	self := quake.Vec3{203.25, 706.75, -759.875}
	goal := quake.Vec3{-148.375, -802.875, -231.875}
	route := []quake.Waypoint{
		{Position: quake.Vec3{174.9, 704, -760}, Kind: 2},
		{Position: quake.Vec3{170, 704, -744}, Kind: 2},
		{Position: quake.Vec3{142.9, 640, -744}, Kind: 2},
		{Position: quake.Vec3{138, 640, -728}, Kind: 2},
		{Position: quake.Vec3{64, 624.9, -728}, Kind: 2},
		{Position: quake.Vec3{64, 620, -727.875}, Kind: 2},
	}
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base3", Frame: 249, Health: 100, OnGround: true, Self: self, SelfVelocity: quake.Vec3{-259.25, 45.125, 0}}, Route: route}, goalPoint: goal}
	if !p.planRampJump() {
		t.Fatal("missing ramp jump")
	}
	if p.jump.phase == 2 {
		t.Fatalf("unsafe direct launch: %+v", *p.jump)
	}
}
