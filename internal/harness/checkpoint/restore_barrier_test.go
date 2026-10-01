package checkpoint

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRestoreBarrierRequiresEveryParticipantAtOneAnchor(t *testing.T) {
	dir := t.TempDir()
	source := &Barrier{ID: "shared", Map: "base2", Frame: 361, Generation: 3}
	captures := []Capture{}
	for i, name := range []string{"Actor", "Bot"} {
		c := Capture{CaptureRequest: CaptureRequest{Version: 1, ID: source.ID, Map: source.Map, Frame: source.Frame, Generation: source.Generation}, Participant: name, SelfEntity: i + 1, Health: 38, Planner: json.RawMessage(`{"version":1,"map":"base2","captured_frame":361}`)}
		if i == 0 {
			c.Runner = json.RawMessage(`{"version":1,"captured_frame":361,"elapsed_frames":61}`)
		}
		captures = append(captures, c)
	}
	proof, err := saveCaptures(dir, source, captures)
	if err != nil {
		t.Fatal(err)
	}
	manifest := Manifest{Map: "base2", Barrier: proof}
	barrier := &Barrier{ID: "shared", Map: "base2", Frame: 1, Generation: 4, Controls: []string{filepath.Join(dir, "actor"), filepath.Join(dir, "bot")}}
	elapsed := 61
	actor := RestoreReceipt{Version: 1, SourceID: "shared", Mode: "resume", Participant: "Actor", SourceFrame: 361, Frame: 1, Generation: 4, SelfEntity: 1, Planner: json.RawMessage(`{"version":1,"map":"base2","captured_frame":1}`), RunnerElapsed: &elapsed}
	bot := actor
	bot.Participant = "Bot"
	bot.SelfEntity = 2
	bot.Mode = "fresh"
	bot.RunnerElapsed = nil
	write := func(r RestoreReceipt, i int) {
		t.Helper()
		if err := WriteCapture(barrier.Controls[i]+".restored.json", r); err != nil {
			t.Fatal(err)
		}
	}
	write(actor, 0)
	write(bot, 1)
	if err := verifyRestoreReceipts(dir, manifest, barrier); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*RestoreReceipt){
		func(r *RestoreReceipt) { r.Frame++ }, func(r *RestoreReceipt) { r.Generation++ }, func(r *RestoreReceipt) { r.SourceID = "other" }, func(r *RestoreReceipt) { r.SourceFrame++ }, func(r *RestoreReceipt) { r.SelfEntity = 2; r.Participant = "Bot" }, func(r *RestoreReceipt) { n := 62; r.RunnerElapsed = &n }, func(r *RestoreReceipt) { r.Planner = json.RawMessage(`{"version":1,"map":"base1","captured_frame":1}`) },
	} {
		bad := actor
		mutate(&bad)
		write(bad, 0)
		if verifyRestoreReceipts(dir, manifest, barrier) == nil {
			t.Fatal("invalid receipt accepted", bad)
		}
	}
	write(actor, 0)
	if err := os.Remove(barrier.Controls[1] + ".restored.json"); err != nil {
		t.Fatal(err)
	}
	if verifyRestoreReceipts(dir, manifest, barrier) == nil {
		t.Fatal("missing participant released")
	}
}
