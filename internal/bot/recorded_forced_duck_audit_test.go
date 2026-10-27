package bot

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"q2coopbot/internal/quake"
)

// Physical recorded-state regression only; this test executes no neural model.
func TestRecordedForcedDuckReleasePathMismatch(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local BSP assets")
	}
	var fixture struct {
		Snapshot quake.Snapshot `json:"snapshot"`
		Proposed quake.UserCmd  `json:"proposed"`
		Applied  quake.UserCmd  `json:"applied"`
	}
	data, err := os.ReadFile("testdata/forced-duck-jail3-frame335.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	g, err := quake.LoadMap(root, fixture.Snapshot.Map)
	if err != nil {
		t.Fatal(err)
	}
	before, _ := json.Marshal(fixture)
	r := DiagnoseRecordedCombatMotion(fixture.Snapshot, fixture.Proposed, fixture.Applied, &g)
	after, _ := json.Marshal(fixture)
	if string(before) != string(after) {
		t.Fatal("diagnostic mutated recorded input")
	}
	if r.Proposed == nil || r.ProposedPathChecked || !r.ForcedDuckObserved || !r.ProposedEffectivePosePathChecked {
		t.Fatal("forced-duck release mismatch not isolated", r)
	}
	crouch := fixture.Proposed
	crouch.Up = -400
	if prediction := predictPmoveGround(fixture.Snapshot, crouch, &g); !reflect.DeepEqual(prediction, r.Proposed) {
		t.Fatal("native-pose prediction differs between forced duck release and explicit crouch")
	}
	for _, up := range []int16{-400, 400} {
		command := fixture.Proposed
		command.Up = up
		if DiagnoseRecordedCombatMotion(fixture.Snapshot, command, fixture.Applied, &g).ForcedDuckObserved {
			t.Fatal("non-release command counted as forced duck", up)
		}
	}
	standing := fixture.Snapshot
	standing.Ducked = false
	if DiagnoseRecordedCombatMotion(standing, fixture.Proposed, fixture.Applied, &g).ForcedDuckObserved {
		t.Fatal("unobserved duck pose inferred")
	}
}
