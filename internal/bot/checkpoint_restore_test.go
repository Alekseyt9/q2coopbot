package bot

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"q2coopbot/internal/harness"
	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/quake"
)

func TestClientCheckpointResumeFreshAndNoEntryReplay(t *testing.T) {
	scene := harness.Scenario{Version: 1, Name: "resume", Map: "base2", StartFrame: 40, GameFrames: 100, Steps: []harness.Step{{ID: "wait", Action: "wait", Frames: 20}}}
	r := harness.New(scene)
	for frame := 40; frame <= 45; frame++ {
		r.Tick(harness.Input{Map: "base2", Frame: frame, Generation: 1, Health: 100, OnGround: true})
	}
	runner, err := r.CaptureCheckpoint()
	if err != nil {
		t.Fatal(err)
	}
	point := quake.Vec3{100, 200, 30}
	state := PlannerCheckpoint{Version: 1, Map: "base2", CapturedFrame: 45, Goal: "return_to_death", DeathPoint: &point, Resources: []CheckpointResource{{Item: quake.Object{ID: 7, Class: "item_health", Origin: point}, Age: 10, Attempted: true}}}
	plannerJSON, _ := json.Marshal(state)
	runnerJSON, _ := json.Marshal(runner)
	source := checkpoint.Capture{CaptureRequest: checkpoint.CaptureRequest{Version: 1, ID: "saved", Map: "base2", Frame: 45, Generation: 1}, Participant: "Bot", SelfEntity: 2, Planner: plannerJSON, Runner: runnerJSON, Health: 100}
	fresh := quake.Snapshot{Map: "base2", Frame: 2, Health: 100, OnGround: true}
	makeClient := func(mode string) *Client {
		d := quake.NewDecoder()
		d.PlayerNumber = 2
		return &Client{name: "Bot", planner: &Planner{}, decoder: d, spawncount: 4, checkpointRestore: &source, checkpointMode: mode, checkpointControl: filepath.Join(t.TempDir(), "control"), scenario: harness.New(scene)}
	}
	c := makeClient("resume")
	if err := c.restoreCheckpointSnapshot(fresh); err != nil {
		t.Fatal(err)
	}
	if !c.checkpointRestored || c.planner.deathPoint == nil || c.planner.resources[7].LastSeen != -8 || !c.planner.resources[7].Attempted {
		t.Fatal("memory not resumed")
	}
	for frame := 2; frame <= 17; frame++ {
		d := c.scenario.Tick(harness.Input{Map: "base2", Frame: frame, Generation: 4, Health: 100, OnGround: true})
		if d.NewStep || d.Place != nil || d.Kill {
			t.Fatal("entry action replayed")
		}
	}
	if c.scenario.Status.State != "completed" || c.scenario.Status.EndFrame != 17 {
		t.Fatal(c.scenario.Status)
	}
	f := makeClient("fresh")
	f.scenario = nil
	if err := f.restoreCheckpointSnapshot(fresh); err != nil {
		t.Fatal(err)
	}
	if !f.checkpointRestored || f.planner.deathPoint != nil || len(f.planner.resources) > 0 {
		t.Fatal("fresh mode imported saved memory")
	}
	for _, change := range []func(*Client){func(c *Client) { c.decoder.PlayerNumber = 1 }, func(c *Client) { c.spawncount = 1 }, func(c *Client) { c.checkpointMode = "unknown" }, func(c *Client) { c.scenario.Scenario.Steps[0].Frames++ }} {
		bad := makeClient("resume")
		change(bad)
		if bad.restoreCheckpointSnapshot(fresh) == nil || bad.checkpointRestored || bad.planner.deathPoint != nil {
			t.Fatal("invalid restore partially applied")
		}
	}
}
