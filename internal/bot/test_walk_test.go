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
