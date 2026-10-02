package bot

import (
	"testing"

	"q2coopbot/internal/quake"
)

func TestWalkScenarioRequiresIsolatedFramePacedActor(t *testing.T) {
	valid := Config{FramePaced: true, Idle: true, TestTeleport: "0,0,24",
		TestWalkTarget: "100,0,24", TestWalkAfterFrames: 25, TestWalkFrames: 30}
	if target, err := validateTestWalk(valid); err != nil || target != (quake.Vec3{100, 0, 24}) {
		t.Fatalf("valid scenario: target=%v err=%v", target, err)
	}
	for _, change := range []func(*Config){
		func(c *Config) { c.FramePaced = false },
		func(c *Config) { c.Idle = false },
		func(c *Config) { c.TestTeleportAfter = "200,0,24" },
		func(c *Config) { c.TestWalkAfterFrames = 1 },
		func(c *Config) { c.TestWalkTarget = "NaN,0,24" },
	} {
		cfg := valid
		change(&cfg)
		if _, err := validateTestWalk(cfg); err == nil {
			t.Fatalf("accepted invalid walking scenario: %+v", cfg)
		}
	}
}

func TestCombatWalkWaitsForReleaseAndUsesReleaseClock(t *testing.T) {
	c := &Client{testCombatBarrier: true, testTeleportSentFrame: 10, testCombatGoFrame: 100}
	if age := c.testWalkAge(130); age != -1 {
		t.Fatalf("walk before barrier: %d", age)
	}
	c.testCombatGo = true
	if age := c.testWalkAge(130); age != 30 {
		t.Fatalf("wrong combat clock: %d", age)
	}
	c.testCombatBarrier = false
	if age := c.testWalkAge(130); age != 120 {
		t.Fatalf("ordinary walk clock changed: %d", age)
	}
}

func TestWalkingPreludeRequiresExplicitActiveRelease(t *testing.T) {
	valid := Config{FramePaced: true, TestWalkThenPlan: true, TestTeleport: "352,1080,-231.875", TestWalkTarget: "480,977,-231.875", TestWalkRoute: true, TestWalkAfterFrames: 25, TestWalkFrames: 80}
	if _, err := validateTestWalk(valid); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Config){
		func(c *Config) { c.Idle = true }, func(c *Config) { c.TestWalkRunIn = true },
		func(c *Config) { c.TestCombatBarrier = true }, func(c *Config) { c.TestWalkTarget = "" },
		func(c *Config) { c.TestWalkFrames = 0 }, func(c *Config) { c.TestTeleportAfter = "0,0,24" },
	} {
		cfg := valid
		change(&cfg)
		if _, err := validateTestWalk(cfg); err == nil {
			t.Fatalf("accepted invalid prelude: %+v", cfg)
		}
	}
}

func TestCombatRunInRequiresBoundedExplicitFixture(t *testing.T) {
	valid := Config{FramePaced: true, TestCombatBarrier: true, TestWalkRunIn: true, TestTeleport: "32,-224,24", TestWalkTarget: "160,-224,24", TestWalkFrames: 3}
	if _, err := validateTestWalk(valid); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Config){
		func(c *Config) { c.TestCombatBarrier = false }, func(c *Config) { c.Idle = true }, func(c *Config) { c.TestWalkFrames = 6 }, func(c *Config) { c.TestWalkAfterFrames = 1 }, func(c *Config) { c.TestWalkRoute = true }, func(c *Config) { c.TestWalkTarget = "" },
	} {
		c := valid
		change(&c)
		if _, err := validateTestWalk(c); err == nil {
			t.Fatalf("accepted invalid run-in: %+v", c)
		}
	}
}

func TestMovementRunInRequiresSettledPlacement(t *testing.T) {
	valid := Config{FramePaced: true, TestWalkRunIn: true, TestTeleport: "764,-391,-63", TestWalkTarget: "750,-385,-63", TestSetupHoldFrames: 6, TestWalkAfterFrames: 6, TestWalkFrames: 1}
	if _, err := validateTestWalk(valid); err != nil {
		t.Fatal(err)
	}
	valid.TestWalkAfterFrames--
	if _, err := validateTestWalk(valid); err == nil {
		t.Fatal("run-in must start after setup hold")
	}
	valid.TestWalkAfterFrames++
	for _, speed := range []int{-1, 301} {
		valid.TestRunInSpeed = speed
		if _, err := validateTestWalk(valid); err == nil {
			t.Fatalf("accepted unbounded run-in speed %d", speed)
		}
	}
}

func TestRoutePreflightReasons(t *testing.T) {
	s := quake.Snapshot{Map: "test", Frame: 40, Health: 100, OnGround: true}
	p := testWalkPath{}
	if cmd := p.command(s, quake.Vec3{}, nil, nil); cmd.Forward != 0 || p.reason != "aas_unavailable" {
		t.Fatal(p.reason, cmd)
	}
	nav := &quake.Navigator{Areas: []quake.Area{{}, {Min: quake.Vec3{-10, -10, -10}, Max: quake.Vec3{10, 10, 10}}}, Edges: make([][]quake.Edge, 2)}
	far := quake.Vec3{10000, 10000, 10000}
	p.command(s, quake.Vec3{11, 0, 0}, nil, nav)
	if p.reason != "route_target_outside_aas" {
		t.Fatal("nearest-area fallback accepted fixture point", p.reason)
	}
	p.command(s, far, nil, nav)
	if p.reason != "route_target_outside_aas" {
		t.Fatal(p.reason)
	}
	s.Self = far
	p.command(s, quake.Vec3{}, nil, nav)
	if p.reason != "route_start_outside_aas" {
		t.Fatal(p.reason)
	}
	s.Self = quake.Vec3{}
	p.command(s, quake.Vec3{}, nil, nav)
	if p.reason != "geometry_unavailable" || !p.ready {
		t.Fatal(p.reason)
	}
}

func TestSearchBaselineStillFollowsVisiblePlayer(t *testing.T) {
	p, s, _ := searchAttemptFixture()
	p.TestDisableSearch, p.searchAttempt, p.probeTarget = true, nil, nil
	p.update(s, "")
	if p.World.Goal != "wait_for_teammate" || p.hasGoal || p.World.SearchAttempt != nil {
		t.Fatalf("baseline searched for hidden player: %+v", p.World)
	}
	visible := quake.Vec3{300, 0, 24}
	s.Frame, s.Teammate = 12, &visible
	p.update(s, "")
	if p.World.Goal != "follow_teammate" || !p.hasGoal {
		t.Fatalf("baseline disabled normal visible following: %+v", p.World)
	}
}

func TestSharedScenarioClockIgnoresClientJoinDelay(t *testing.T) {
	a := Client{testTeleportSentFrame: 25, testScenarioFrameOrigin: 40}
	b := Client{testTeleportSentFrame: 29, testScenarioFrameOrigin: 40}
	if a.testScenarioAge(65) != 25 || b.testScenarioAge(65) != 25 {
		t.Fatal("paired movement shifted with client join timing")
	}
	a.testScenarioFrameOrigin = 0
	if a.testScenarioAge(65) != 40 {
		t.Fatal("legacy teleport-relative clock changed")
	}
}

func TestRoutedWalkKeepsProgressAndResetsOnNewEpisode(t *testing.T) {
	p := testWalkPath{ready: true, mapName: "test", lastFrame: 10, route: []quake.Waypoint{
		{Position: quake.Vec3{10, 0, 24}}, {Position: quake.Vec3{100, 0, 24}},
	}}
	s := quake.Snapshot{Map: "test", Frame: 11, Self: quake.Vec3{10, 0, 24}}
	nav := &quake.Navigator{}
	p.command(s, quake.Vec3{200, 0, 24}, nil, nav)
	if p.next != 1 {
		t.Fatal("completed waypoint retained")
	}
	s.Frame = 12
	s.Self = quake.Vec3{-30, 0, 24}
	p.command(s, quake.Vec3{200, 0, 24}, nil, nav)
	if p.next != 1 {
		t.Fatal("route progress moved backwards")
	}
	s.Frame = 1
	p.command(s, quake.Vec3{200, 0, 24}, nil, nav)
	if p.ready || p.next != 0 {
		t.Fatal("new episode reused old route")
	}
	if _, err := validateTestWalk(Config{TestWalkRoute: true}); err == nil {
		t.Fatal("route mode without target accepted")
	}
}
