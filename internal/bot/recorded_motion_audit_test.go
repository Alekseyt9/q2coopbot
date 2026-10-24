package bot

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"

	"q2coopbot/internal/quake"
)

func TestRecordedMotionAuditMissingGeometry(t *testing.T) {
	s := quake.Snapshot{OnGround: true, Health: 100}
	r := DiagnoseRecordedCombatMotion(s, quake.UserCmd{Up: -400}, quake.UserCmd{}, nil)
	if r.StoppedCrouchEligible || r.StoppedCrouchPathChecked || r.Applied != nil || r.Proposed != nil {
		t.Fatal("missing geometry admitted a counterfactual", r)
	}
}

func TestRecordedMotionAuditPreservesInputsAndScope(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local BSP assets")
	}
	var recorded struct {
		Snapshot quake.Snapshot
		Command  quake.UserCmd
	}
	b, err := os.ReadFile("testdata/pmove-jail1-door-frame194.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &recorded); err != nil {
		t.Fatal(err)
	}
	g, err := quake.LoadMap(root, recorded.Snapshot.Map)
	if err != nil {
		t.Fatal(err)
	}
	s := recorded.Snapshot
	before, _ := json.Marshal(s)
	applied := recorded.Command
	applied.Forward, applied.Side, applied.Up = 0, 0, 0
	proposed := recorded.Command
	proposed.Up = -400
	originalApplied, originalProposed := applied, proposed
	r := DiagnoseRecordedCombatMotion(s, proposed, applied, &g)
	after, _ := json.Marshal(s)
	if !bytes.Equal(before, after) || applied != originalApplied || proposed != originalProposed {
		t.Fatal("offline diagnostic changed recorded inputs")
	}
	expected := applied
	expected.Up = proposed.Up
	if !r.StoppedCrouchEligible || r.StoppedCrouchCommand != expected {
		t.Fatal("counterfactual changed aim/fire/horizontal command", r.StoppedCrouchCommand, expected)
	}
	for _, change := range []struct {
		s quake.Snapshot
		p quake.UserCmd
		a quake.UserCmd
	}{
		{func() quake.Snapshot { v := s; v.OnGround = false; return v }(), proposed, applied},
		{s, func() quake.UserCmd { v := proposed; v.Up = 400; return v }(), applied},
		{s, proposed, func() quake.UserCmd { v := applied; v.Forward = 1; return v }()},
		{s, proposed, func() quake.UserCmd { v := applied; v.Side = 1; return v }()},
		{s, proposed, func() quake.UserCmd { v := applied; v.Up = -400; return v }()},
	} {
		if DiagnoseRecordedCombatMotion(change.s, change.p, change.a, &g).StoppedCrouchEligible {
			t.Fatal("unsupported or already-applied pose admitted", change)
		}
	}
}
