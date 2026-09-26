package bot

import (
	"encoding/json"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness"
	"testing"
)

func TestScenarioStopRequiresTailAndServerAck(t *testing.T) {
	path := filepath.Join(t.TempDir(), "completion.json")
	c := Client{scenarioResultPath: path, scenarioTailFrames: 5, spawncount: 7, lastMoveFrame: 44, latestFrame: 45}
	c.planner = &Planner{}
	c.planner.World.Map = "base2"
	if stop, err := c.scenarioShouldStop(); stop || err != nil {
		t.Fatal(stop, err)
	}
	data, _ := json.Marshal(scenarioCompletion{Map: "base2", Generation: 7, EndFrame: 44, State: "failed"})
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if stop, err := c.scenarioShouldStop(); stop || err != nil {
		t.Fatal(stop, err)
	}
	c.lastMoveFrame = 49
	c.latestFrame = 49
	if stop, _ := c.scenarioShouldStop(); stop {
		t.Fatal("unacknowledged last command")
	}
	c.latestFrame = 50
	if stop, err := c.scenarioShouldStop(); !stop || err != nil {
		t.Fatal(stop, err)
	}
	c.spawncount = 8
	if _, err := c.scenarioShouldStop(); err == nil {
		t.Fatal("accepted stale generation")
	}
}

func TestScenarioCompletionPublishedOnceAfterTerminalState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "completion.json")
	c := Client{scenarioResultPath: path, spawncount: 7, planner: &Planner{}, scenario: &harness.Runner{Status: harness.Status{State: "running"}}}
	c.planner.World.Map = "base2"
	if err := c.publishScenarioCompletion(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("running scenario published completion")
	}
	c.scenario.Status = harness.Status{State: "completed", EndFrame: 44}
	if err := c.publishScenarioCompletion(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var end scenarioCompletion
	if err = json.Unmarshal(data, &end); err != nil {
		t.Fatal(err)
	}
	if end.Map != "base2" || end.Generation != 7 || end.EndFrame != 44 {
		t.Fatal(end)
	}
	c.scenario.Status.EndFrame = 99
	if err = c.publishScenarioCompletion(); err != nil {
		t.Fatal(err)
	}
	data, _ = os.ReadFile(path)
	_ = json.Unmarshal(data, &end)
	if end.EndFrame != 44 {
		t.Fatal("terminal evidence changed")
	}
	if _, err = os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatal("temporary completion left behind")
	}
}
