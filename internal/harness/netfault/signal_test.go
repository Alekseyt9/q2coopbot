package netfault

import (
	"encoding/json"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness/coord"
	"testing"
)

func TestStateSignalIdentityAndFreshness(t *testing.T) {
	path := filepath.Join(t.TempDir(), "arm.json")
	f := GameFrame{Map: "base2", Generation: 123, Frame: 100}
	if r, err := signalAt(path, f); err != nil || r != nil {
		t.Fatal("missing signal must wait", r, err)
	}
	for _, name := range []string{"ready", "future", "map", "generation", "phase", "role", "stale", "invalid-frame"} {
		t.Run(name, func(t *testing.T) {
			r := coord.Ready{Map: f.Map, Generation: f.Generation, Frame: 98, Role: "observer"}
			switch name {
			case "future":
				r.Frame = 101
			case "map":
				r.Map = "base1"
			case "generation":
				r.Generation++
			case "phase":
				r.Phase = 1
			case "role":
				r.Role = "actor"
			case "stale":
				r.Frame = 79
			case "invalid-frame":
				r.Frame = 0
			}
			data, _ := json.Marshal(r)
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			got, err := signalAt(path, f)
			if name == "ready" {
				if err != nil || got == nil {
					t.Fatal(got, err)
				}
			} else if name == "future" {
				if err != nil || got != nil {
					t.Fatal(got, err)
				}
			} else if err == nil {
				t.Fatal("invalid trigger accepted")
			}
		})
	}
	for _, c := range []Config{
		{Listen: "127.0.0.1:1", Server: "127.0.0.1:2", DurationMS: 1, ArmSignalPath: path},
		{Listen: "127.0.0.1:1", Server: "127.0.0.1:2", DurationMS: 1, DecodeQuake: true, ArmSignalPath: path, ArmBarrierDir: "barrier"},
	} {
		if c.Validate() == nil {
			t.Fatal("ambiguous/unobservable trigger accepted")
		}
	}
}
