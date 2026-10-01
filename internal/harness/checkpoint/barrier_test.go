package checkpoint

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestBarrierCaptureAnchorsAndImmutableFiles(t *testing.T) {
	r := CaptureRequest{Version: 1, ID: "capture_a", Map: "base2", Frame: 100, Generation: 3}
	c := Capture{CaptureRequest: r, Participant: "Bot", Health: 38, Planner: json.RawMessage(`{"version":1,"map":"base2","captured_frame":100}`), Runner: json.RawMessage(`{"version":1,"captured_frame":100}`)}
	if err := validateCapture(c, r); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*Capture){func(c *Capture) { c.Frame++ }, func(c *Capture) { c.Generation++ }, func(c *Capture) { c.Error = "capture failed" }, func(c *Capture) { c.Planner = json.RawMessage(`{"version":1,"map":"base2","captured_frame":101}`) }, func(c *Capture) { c.Runner = json.RawMessage(`{"version":1,"captured_frame":99}`) }, func(c *Capture) { c.Participant = "../outside" }} {
		bad := c
		change(&bad)
		if validateCapture(bad, r) == nil {
			t.Fatal("bad anchor accepted", bad)
		}
	}
	root := t.TempDir()
	b := &Barrier{ID: r.ID, Map: r.Map, Frame: r.Frame, Generation: r.Generation, Controls: []string{filepath.Join(root, "control")}}
	if b.validate() != nil {
		t.Fatal(b)
	}
	proof, err := saveCaptures(root, b, []Capture{c})
	if err != nil {
		t.Fatal(err)
	}
	if len(proof.Participants) != 1 {
		t.Fatal(proof)
	}
	if err = verifySavedCaptures(root, r.Map, proof); err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*BarrierProof){
		func(p *BarrierProof) { p.Frame++ },
		func(p *BarrierProof) { p.Generation++ },
		func(p *BarrierProof) { p.Participants = nil },
		func(p *BarrierProof) { p.Participants = append(p.Participants, p.Participants[0]) },
	} {
		bad := *proof
		change(&bad)
		if verifySavedCaptures(root, r.Map, &bad) == nil {
			t.Fatal("invalid saved proof accepted", bad)
		}
	}
	file := filepath.Join(root, "sidecar", "Bot.json")
	before := proof.Participants[0]
	if err = os.WriteFile(file, []byte("corrupted"), 0600); err != nil {
		t.Fatal(err)
	}
	after, err := fileRecord(filepath.Dir(file), "Bot.json")
	if err != nil || after == before {
		t.Fatal("integrity change missed")
	}
	if verifySavedCaptures(root, r.Map, proof) == nil {
		t.Fatal("corrupted sidecar accepted")
	}
	b.Controls = append(b.Controls, b.Controls[0])
	if b.validate() == nil {
		t.Fatal("duplicate controls")
	}
}

func TestParticipantBindingsRejectAmbiguousNativeState(t *testing.T) {
	root := t.TempDir()
	b := &Barrier{ID: "binding", Map: "base2", Frame: 100, Generation: 3}
	c := Capture{CaptureRequest: CaptureRequest{Version: 1, ID: b.ID, Map: b.Map, Frame: b.Frame, Generation: b.Generation}, Participant: "Bot", SelfEntity: 2, Health: 38, Planner: json.RawMessage(`{"version":1,"map":"base2","captured_frame":100}`)}
	proof, err := saveCaptures(root, b, []Capture{c})
	if err != nil {
		t.Fatal(err)
	}
	bindings, err := participantBindings(root, proof)
	if err != nil || bindings[2] != "Bot" {
		t.Fatal(bindings, err)
	}
	if err := WriteCapture(filepath.Join(root, "manifest.json"), Manifest{Version: 1, Map: b.Map, Barrier: proof}); err != nil {
		t.Fatal(err)
	}
	loaded, err := ReadParticipant(root, "Bot")
	if err != nil || loaded.SelfEntity != 2 || loaded.Health != 38 || loaded.Participant != "Bot" {
		t.Fatal("participant payload lost", loaded, err)
	}
	for _, slot := range []int{0, -1, 257} {
		bad := c
		bad.SelfEntity = slot
		if err := WriteCapture(filepath.Join(root, "sidecar/Bot.json"), bad); err != nil {
			t.Fatal(err)
		}
		if _, err := participantBindings(root, proof); err == nil {
			t.Fatal("invalid slot accepted", slot)
		}
	}
	if _, err := participantBindings(root, nil); err == nil {
		t.Fatal("native-only binding accepted")
	}
	if err := WriteCapture(filepath.Join(root, "sidecar/Bot.json"), c); err != nil {
		t.Fatal(err)
	}
	duplicate := c
	duplicate.Participant = "Actor"
	if err := WriteCapture(filepath.Join(root, "sidecar/Actor.json"), duplicate); err != nil {
		t.Fatal(err)
	}
	proof.Participants = append(proof.Participants, File{Name: "Actor.json"})
	if _, err := participantBindings(root, proof); err == nil {
		t.Fatal("two participants in one native slot accepted")
	}
}
