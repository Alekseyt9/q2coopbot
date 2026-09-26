package bot

import (
	"encoding/json"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness"
	"q2coopbot/internal/quake"
	"testing"
	"time"
)

func botSessionFixture() harness.Session {
	a := harness.Scenario{Version: 1, Name: "first", Map: "base1", StartFrame: 40, GameFrames: 100, ActorOrigin: quake.Vec3{1, 2, 3}, BotOrigin: quake.Vec3{4, 5, 6}, Steps: []harness.Step{{ID: "wait", Action: "wait", Frames: 2}}}
	b := a
	b.Map = "base2"
	b.ActorOrigin = quake.Vec3{7, 8, 9}
	return harness.Session{Version: 1, Name: "two maps", TransitionTimeoutMS: 1000, Phases: []harness.Phase{{ID: "a", Scenario: a}, {ID: "b", Scenario: b}}}
}

func TestConfigureSessionRoles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.json")
	data, _ := json.Marshal(botSessionFixture())
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	for _, role := range []string{"actor", "observer"} {
		cfg := Config{TestSession: path, TestSessionRole: role, FramePaced: true, Duration: time.Minute, Idle: role == "actor", TestRCONPassword: "test"}
		s, r, err := configureSession(&cfg)
		if err != nil || s == nil || (r != nil) != (role == "actor") || cfg.TestTeleportMap != "base1" {
			t.Fatal(s, r, err)
		}
		want := "4,5,6"
		if role == "actor" {
			want = "1,2,3"
		}
		if cfg.TestTeleport != want {
			t.Fatal(cfg.TestTeleport)
		}
	}
	for _, mutate := range []func(*Config){func(c *Config) { c.GameFrames = 100 }, func(c *Config) { c.Idle = false }, func(c *Config) { c.TestRCONPassword = "" }, func(c *Config) { c.ExitOnReconnect = true }, func(c *Config) { c.TestScenario = "old.json" }} {
		cfg := Config{TestSession: path, TestSessionRole: "actor", FramePaced: true, Duration: time.Minute, Idle: true, TestRCONPassword: "test"}
		mutate(&cfg)
		if _, _, err := configureSession(&cfg); err == nil {
			t.Fatal("conflicting config accepted")
		}
	}
}

func TestSessionPlacementOncePerGeneration(t *testing.T) {
	s := botSessionFixture()
	r, _ := harness.NewSession(s)
	c := Client{sessionDefinition: &s, session: r, begun: true, frameReady: true, spawncount: 2, latestFrame: 32, planner: &Planner{}}
	c.planner.World.Map = "base1"
	if err := c.prepareSessionPhase(); err != nil {
		t.Fatal(err)
	}
	if c.testTeleportPosition != s.Phases[0].Scenario.ActorOrigin {
		t.Fatal(c.testTeleportPosition)
	}
	c.testTeleportSent = true
	if err := c.prepareSessionPhase(); err != nil || !c.testTeleportSent {
		t.Fatal("repeated placement", err)
	}
	c.spawncount = 3
	c.planner.World.Map = "base2"
	if err := c.prepareSessionPhase(); err != nil {
		t.Fatal(err)
	}
	if c.testTeleportSent || c.sessionPhase != 1 || c.testTeleportPosition != s.Phases[1].Scenario.ActorOrigin {
		t.Fatal("new phase did not reset setup")
	}
	c.spawncount = 4
	if err := c.prepareSessionPhase(); err == nil {
		t.Fatal("unexpected extra generation accepted")
	}
}

func TestSessionIntermediateCompletionNotPublished(t *testing.T) {
	s := botSessionFixture()
	r, _ := harness.NewSession(s)
	path := filepath.Join(t.TempDir(), "completion.json")
	c := Client{session: r, scenarioResultPath: path, spawncount: 2, planner: &Planner{}}
	c.planner.World.Map = "base1"
	for f := 40; f <= 42; f++ {
		r.Tick(harness.Input{Map: "base1", Generation: 2, Frame: f, Health: 100}, time.Duration(f)*time.Millisecond)
	}
	if r.Status.State != "waiting_map" {
		t.Fatal(r.Status)
	}
	if err := c.publishScenarioCompletion(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("intermediate phase stopped whole session")
	}
	c.spawncount = 3
	c.planner.World.Map = "base2"
	for f := 40; f <= 42; f++ {
		r.Tick(harness.Input{Map: "base2", Generation: 3, Frame: f, Health: 100}, time.Duration(f+10)*time.Millisecond)
	}
	if err := c.publishScenarioCompletion(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var end scenarioCompletion
	if err := json.Unmarshal(data, &end); err != nil {
		t.Fatal(err)
	}
	if end.Map != "base2" || end.Generation != 3 || end.EndFrame != 42 || end.State != "completed" {
		t.Fatal(end)
	}
}
