package netfault

import (
	"encoding/json"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness/coord"
	"testing"
)

func TestBarrierRequiresBothRolesAndExactStart(t *testing.T) {
	c := Config{ArmBarrierDir: t.TempDir(), ArmPhase: 0}
	f := GameFrame{Map: "base1", Generation: 7, Frame: 50}
	write := func(role string, r coord.Ready) {
		t.Helper()
		data, _ := json.Marshal(r)
		if err := os.WriteFile(filepath.Join(c.ArmBarrierDir, "0-7-"+role+".json"), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	a := coord.Ready{Map: "base1", Generation: 7, Phase: 0, Frame: 38, Role: "actor"}
	b := a
	b.Role = "observer"
	b.Frame = 40
	write("actor", a)
	if e, err := barrierAt(c, f); e != nil || err != nil {
		t.Fatal(e, err)
	}
	write("observer", b)
	f.Frame = 49
	if e, err := barrierAt(c, f); e != nil || err != nil {
		t.Fatal(e, err)
	}
	f.Frame = 50
	e, err := barrierAt(c, f)
	if e == nil || err != nil {
		t.Fatal(e, err)
	}
	if err := verifyBarrier(c, Event{Barrier: e, Frames: []GameFrame{f}}); err != nil {
		t.Fatal(err)
	}
	f.Frame = 51
	if _, err := barrierAt(c, f); err == nil {
		t.Fatal("late trigger accepted")
	}
	f.Frame = 50
	b.Map = "base2"
	write("observer", b)
	if _, err := barrierAt(c, f); err == nil {
		t.Fatal("foreign readiness accepted")
	}
}

func TestReportRequiresRecordedBarrierEvidence(t *testing.T) {
	c, rows := reportFixture()
	c.DecodeQuake = true
	c.ArmBarrierDir = "barrier"
	e := &BarrierEvidence{Actor: coord.Ready{Map: "base1", Generation: 7, Frame: 40, Role: "actor"}, Observer: coord.Ready{Map: "base1", Generation: 7, Frame: 40, Role: "observer"}}
	rows[1].Barrier = e
	rows[1].Frames = []GameFrame{{Map: "base1", Generation: 7, Frame: 50}}
	if r := Analyze(c, rows, 20, true); !r.Accepted {
		t.Fatal(r)
	}
	rows[1].Barrier = nil
	if r := Analyze(c, rows, 20, true); r.Accepted {
		t.Fatal("missing barrier accepted")
	}
	rows[1].Barrier = e
	e.Observer.Frame = 41
	if r := Analyze(c, rows, 20, true); r.Accepted {
		t.Fatal("wrong start accepted")
	}
}
