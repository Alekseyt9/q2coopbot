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

func TestBase3RampLandingWalkable(t *testing.T) {
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
	from := quake.Vec3{107, 634.5, -727.625}
	to := quake.Vec3{64, 624.9, -727.875}
	p := &Planner{Nav: n, World: World{Geometry: &g}}
	if !p.jumpLandingOK(from, to) {
		t.Fatal("supported ramp landing was counted as a miss")
	}
	if !p.jumpLandingOK(quake.Vec3{116.5, 626.875, -727.625}, to) {
		t.Fatal("early supported ramp landing was counted as a miss")
	}
	if p.jumpLandingOK(quake.Vec3{130, 634.5, -727.625}, to) {
		t.Fatal("far landing was counted as safe")
	}
	if p.jumpLandingOK(quake.Vec3{107, 634.5, -760}, to) {
		t.Fatal("lower landing was counted as safe")
	}
}

func TestBase3FirstLipKeepsApproachMomentum(t *testing.T) {
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
	s := quake.Snapshot{Map: "base3", Frame: 113, Health: 100, OnGround: true,
		Self: quake.Vec3{1503.375, 1397, -804.875}, SelfVelocity: quake.Vec3{-112.375, -42, 21.125}}
	goal := quake.Vec3{-148.375, -802.875, -231.875}
	p := &Planner{Nav: n, World: World{Map: s.Map, Geometry: &g, GeometryStatus: "ready", Snapshot: s, Goal: "regroup_after_respawn"}, goalPoint: goal}
	p.World.Route, _ = n.Route(s.Self, goal)
	if !p.planGapJump() || p.jump.phase != 2 || p.jump.runup != (quake.Vec3{}) {
		t.Fatalf("expected direct first-lip takeoff, got %+v", p.jump)
	}
}

func TestBase1CornerLoopEscape(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base1.aas")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Map: "base1", Frame: 230, Health: 100, OnGround: true, Self: quake.Vec3{-21.25, -565, -79.875}}
	goal := quake.Vec3{960, 408, -167.875}
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: s}, goalPoint: goal}
	p.World.Route, _ = n.Route(s.Self, goal)
	for i := 0; i < 4; i++ {
		p.cornerHistory = append(p.cornerHistory,
			quake.Vec3{-31.25, -570.75, -79.875},
			quake.Vec3{-28.125, -569.375, -79.875},
			quake.Vec3{-21.25, -565, -79.875})
	}
	target, ok := p.regroupCornerEscape(s)
	if !ok || target[0] < 96 || p.cornerEscapeUntil != s.Frame+12 {
		t.Fatalf("no bounded escape toward the later route: %v %v", target, ok)
	}
	if p.cornerEscapeApproachClear(quake.Snapshot{Self: quake.Vec3{-43.875, -596.375, -79.75}}, target) {
		t.Fatal("early bypass enters a wall after the first clear step")
	}
	s.Frame++
	s.Self = quake.Vec3{11, -565, -79.875}
	if next, active := p.regroupCornerEscape(s); !active || next != target {
		t.Fatalf("escape was not held across replan: %v %v", next, active)
	}
	s.Frame += 13
	if _, active := p.regroupCornerEscape(s); active {
		t.Fatal("expired corner override remained active")
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
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base3", Frame: 100, Health: 100, Gravity: 800, OnGround: true, Self: self}, Route: route}, goalPoint: goal}
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

func TestBase1FarReturnFirstRise(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base1.aas")
	if err != nil {
		t.Fatal(err)
	}
	self := quake.Vec3{747.5, -408.25, -71.875}
	goal := quake.Vec3{960, 408, -167.875}
	route, ok := n.Route(self, goal)
	if !ok {
		t.Fatal("missing AAS route")
	}
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base1", Frame: 301, Health: 100, OnGround: true, Self: self}, Route: route}, goalPoint: goal}
	if !p.blockedRiseApproach() || !p.planRampJump() {
		t.Fatal("first rise needs a verified landing")
	}
	landing := p.jump.landing
	if landing[0] != 769 || landing[2] <= -41 || !g.PlayerMoveClear(landing, landing) || !n.GroundedNear(landing) {
		t.Fatalf("invalid corrected landing: %+v", landing)
	}
	// The failed native approach must align on supported ground, rather than
	// launch the standing arc while still moving west.
	p.World.Snapshot.Self = quake.Vec3{741.875, -410.5, -71.875}
	p.World.Snapshot.SelfVelocity = quake.Vec3{-55.125, -23.5, 0}
	p.World.Route, _ = n.Route(p.World.Snapshot.Self, goal)
	if !p.planRampJump() || p.jump.phase != 4 || p.jump.runup[0] <= p.World.Snapshot.Self[0] {
		t.Fatalf("missing supported alignment: %+v", p.jump)
	}
	if g.GroundMoveHazardStep(n, p.World.Snapshot.Self, p.jump.runup[0]-p.World.Snapshot.Self[0], p.jump.runup[1]-p.World.Snapshot.Self[1], 6) != "" {
		t.Fatal("alignment crossed unsupported ground")
	}
	// Native physics can land on the slope before the AAS reach. The short
	// continuation still needs a checked arc and a route beyond the rise.
	p.World.Snapshot.Self = quake.Vec3{774, -360, -47.5}
	p.World.Snapshot.SelfVelocity = quake.Vec3{}
	p.World.Route, _ = n.Route(p.World.Snapshot.Self, goal)
	if !p.planRampJump() || p.jump.phase != 2 || !p.jump.steerVelocity || p.jump.landing[2] <= -41 {
		t.Fatalf("missing verified continuation: %+v", p.jump)
	}
}

func TestBase1SecondRiseEntryHasVerifiedShortJump(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base1.aas")
	if err != nil {
		t.Fatal(err)
	}
	goal := quake.Vec3{960, 408, -167.875}
	for _, self := range []quake.Vec3{{815.5, -324.875, -29}, {815.75, -325.125, -29.25}} {
		s := quake.Snapshot{Map: "base1", Frame: 100, Health: 100, Gravity: 800, OnGround: true, Self: self}
		p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: s}, goalPoint: goal}
		p.World.Route, _ = n.Route(self, goal)
		target := p.World.Route[0].Position
		if g.GroundMoveHazardStep(n, self, target[0]-self[0], target[1]-self[1], 8) == "" {
			t.Fatal("fixture no longer reproduces blocked direct approach")
		}
		if !p.planRampJump() || p.jump.phase != 2 || !p.jump.steerVelocity || quake.Horizontal(self, p.jump.landing) >= 48 {
			t.Fatalf("missing verified short jump from %v: %+v", self, p.jump)
		}
		if !g.PlayerMoveClear(p.jump.landing, p.jump.landing) || !n.GroundedNear(p.jump.landing) {
			t.Fatal("short jump has no clear grounded landing")
		}
		if _, ok := n.Route(p.jump.landing, goal); !ok {
			t.Fatal("short jump has no route beyond landing")
		}
	}
}

func TestBase1ReturnDropTargetClearsUpperPlatformEdge(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base1.aas")
	if err != nil {
		t.Fatal(err)
	}
	self, goal := quake.Vec3{963.5, 55.25, -31.875}, quake.Vec3{960, 408, -167.875}
	s := quake.Snapshot{Map: "base1", Frame: 100, Health: 100, Gravity: 800, OnGround: true, Self: self}
	p := &Planner{Nav: n, World: World{Geometry: &g, Snapshot: s, Goal: "regroup_after_respawn"}, goalPoint: goal}
	p.World.Route, _ = n.Route(self, goal)
	if !p.planWalkOff() || !p.jump.drop || p.jump.landing[1] <= 113.5 {
		t.Fatalf("drop still aims at the upper platform boundary: %+v", p.jump)
	}
	landing := p.jump.landing
	above := landing
	above[2] = self[2]
	if !g.PlayerMoveClear(self, above) || !g.PlayerMoveClear(above, landing) {
		t.Fatal("edge margin enters static geometry")
	}
	if _, ok := n.Route(landing, goal); !ok {
		t.Fatal("edge margin has no onward route")
	}
}

func TestBase1EarlyRampAlignsWithoutRetreat(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("assets")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(root + "/maps/base1.aas")
	if err != nil {
		t.Fatal(err)
	}
	self, goal := quake.Vec3{669.625, -457.125, -95.875}, quake.Vec3{960, 408, -167.875}
	s := quake.Snapshot{Map: "base1", Frame: 100, Health: 100, Gravity: 800, OnGround: true, Self: self, SelfVelocity: quake.Vec3{92.75, 27.375, 0}}
	p := &Planner{Nav: n, World: World{Geometry: &g, Snapshot: s, Goal: "regroup_after_respawn"}, goalPoint: goal}
	p.World.Route, _ = n.Route(self, goal)
	if !p.planRampJump() || p.jump.phase != 4 || !p.jump.steerVelocity || p.jump.runup[0] <= self[0] || quake.Horizontal(self, p.jump.landing) > 80 {
		t.Fatalf("early rise still retreats to an unsafe run-up: %+v", p.jump)
	}
	if g.GroundMoveHazardStep(n, self, p.jump.runup[0]-self[0], p.jump.runup[1]-self[1], 6) != "" {
		t.Fatal("alignment is not grounded")
	}
	// Revalidate the native end of alignment before launching, without
	// returning to the old launch point or forgetting measured velocity.
	p.World.Snapshot.Frame++
	p.World.Snapshot.Self = quake.Vec3{675.5, -455.875, -95.875}
	p.World.Snapshot.SelfVelocity = quake.Vec3{60, 11.375, 0}
	cmd, active := p.jumpCommand(quake.UserCmd{})
	if !active || cmd.Up == 0 || p.jump == nil || p.jump.phase != 2 || !p.jump.steerVelocity {
		t.Fatalf("alignment did not produce a revalidated takeoff: %+v %+v", cmd, p.jump)
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
	s := quake.Snapshot{Map: "base3", Frame: 100, Health: 100, Gravity: 800, OnGround: true, Self: self}
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
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base3", Frame: 100, Health: 100, Gravity: 800, OnGround: true, Self: self, SelfVelocity: quake.Vec3{-188, -242, 0}}, Route: route}, goalPoint: goal}
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
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base3", Frame: 100, Health: 100, Gravity: 800, OnGround: true, Self: self}, Route: route}, goalPoint: goal}
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

func TestBase3ShortRampStandingAcceleration(t *testing.T) {
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
	self := quake.Vec3{159.25, 665.375, -743.875}
	goal := quake.Vec3{-148.375, -802.875, -231.875}
	route, ok := n.Route(self, goal)
	if !ok {
		t.Fatal("missing route")
	}
	p := &Planner{Nav: n, World: World{Geometry: &g, Goal: "regroup_after_respawn", Snapshot: quake.Snapshot{Map: "base3", Frame: 244, Health: 100, OnGround: true, Self: self, SelfVelocity: quake.Vec3{-54.5, -11.5, 0}}, Route: route}, goalPoint: goal}
	if !p.planRampJump() || p.jump.phase != 2 || p.jump.speed != 400 {
		t.Fatalf("standing ramp needs accelerated launch: %+v", p.jump)
	}
}
