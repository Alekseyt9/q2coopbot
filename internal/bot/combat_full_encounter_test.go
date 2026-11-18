package bot

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func fullEncounterModel(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile(combatCheckpointModel(t))
	if err != nil {
		t.Fatal(err)
	}
	var f policy.HJEPAFile
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	f.OwnershipScope = policy.HJEPAFullEncounterScope
	data, err = json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "full.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestFullEncounterOwnsColdSearchOcclusionAndObservedDefeat(t *testing.T) {
	c := policyClient(t, "learned")
	p, err := policy.LoadHJEPA(fullEncounterModel(t))
	if err != nil {
		t.Fatal(err)
	}
	c.combatControl.provider = p
	c.testSynchronous = true
	clear := true
	for frame := 20; frame < 650; frame++ {
		c.latestFrame = frame
		s := &c.planner.World.Snapshot
		s.Frame, s.Enemies, s.Defeated = frame, nil, nil
		if frame == 270 {
			s.Enemies = []quake.Object{{ID: 300, Class: "monster_soldier", Origin: quake.Vec3{200, -224, 24}, ClearShot: &clear}}
		}
		if frame == 310 {
			s.Defeated = []quake.Object{{ID: 300, Class: "monster_soldier"}}
		}
		c.planner.World.Updated = time.Now()
		o := c.combatObservation(time.Now())
		_, _, selection, direct := c.combatCommand(o, time.Now())
		if !direct || selection.Owner != "provider" || selection.Fallback != "" || selection.CombatContinuation != (len(o.Enemies) == 0) {
			t.Fatalf("lost ownership at %d: %+v", frame, selection)
		}
		if frame != 270 && (len(o.Enemies) != 0 || selection.Candidate.TargetEntity != 0) {
			t.Fatal("invented an unseen target")
		}
	}
	for _, invalid := range []string{"dead", "stale", "setup"} {
		o := c.combatObservation(time.Now())
		c.planner.testSetupHold = invalid == "setup"
		if invalid == "dead" {
			o.Health = 0
		}
		if invalid == "stale" {
			o.AgeMS = 301
		}
		if _, _, s, direct := c.combatCommand(o, time.Now()); direct || s.Owner != "rules" || s.Fallback == "" {
			t.Fatal("scope bypassed safety", invalid, s)
		}
	}
}

func TestFullEncounterCheckpointAfterLongEmptyPrefix(t *testing.T) {
	path := fullEncounterModel(t)
	source := checkpointTestClient(t, path, 2, 3)
	observe := func(c *Client, frame int) policy.Observation {
		c.latestFrame = frame
		c.planner.World.Snapshot = checkpointTestSnapshot(frame, 7)
		c.planner.World.Snapshot.Enemies = nil
		c.planner.World.Updated = time.Now()
		o := c.combatObservation(time.Now())
		c.combatControl.engagement.observe(o, nil)
		return o
	}
	for frame := 170; frame < 510; frame++ {
		if _, err := source.combatControl.provider.Decide(observe(source, frame)); err != nil {
			t.Fatal(err)
		}
	}
	root := source.combatControl.last
	root.Frame++
	state, err := source.combatControl.captureCombatCheckpoint(root, source.previous)
	if err != nil {
		t.Fatal(err)
	}
	if state.Engagement.LastSeen != 0 || len(state.Engagement.Known) != 0 {
		t.Fatal("empty history fabricated contact")
	}
	native := root
	native.Frame, native.Connection, native.Spawncount = 1, 4, 55
	wrong := checkpointTestClient(t, combatCheckpointModel(t), 4, 55)
	if _, err := wrong.combatControl.restoreCombatCheckpoint(state, native); err == nil {
		t.Fatal("legacy scope restored full-encounter checkpoint")
	}
	loaded := checkpointTestClient(t, path, 4, 55)
	if _, err := loaded.combatControl.restoreCombatCheckpoint(state, native); err != nil {
		t.Fatal(err)
	}
	for step := 0; step < 12; step++ {
		a, b := observe(source, root.Frame+step), observe(loaded, 1+step)
		x, _ := policy.HJEPAFeatures(a, policy.HJEPAContextFeatureVersion)
		y, _ := policy.HJEPAFeatures(b, policy.HJEPAContextFeatureVersion)
		if !reflect.DeepEqual(x, y) {
			t.Fatal("empty-prefix features changed on restore", step)
		}
		want, e1 := source.combatControl.provider.Decide(a)
		got, e2 := loaded.combatControl.provider.Decide(b)
		want.Identity = got.Identity
		if e1 != nil || e2 != nil || want != got || !reflect.DeepEqual(source.combatControl.provider.(*policy.HJEPA).LastSample(), loaded.combatControl.provider.(*policy.HJEPA).LastSample()) {
			t.Fatal("restored empty search diverged", step, e1, e2)
		}
	}
	nestedRoot := loaded.combatControl.last
	nestedRoot.Frame++
	if _, err := loaded.combatControl.captureCombatCheckpoint(nestedRoot, loaded.previous); err != nil {
		t.Fatal("nested empty checkpoint", err)
	}
}
