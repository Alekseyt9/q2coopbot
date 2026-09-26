package harness

import (
	"q2coopbot/internal/quake"
	"testing"
)

func cycleFixture() []Trace {
	var rows []Trace
	for frame := 50; frame <= 58; frame++ {
		health := int16(100)
		if frame >= 52 && frame <= 54 {
			health = 0
		}
		row := Trace{Frame: frame, Health: &health, ObserverKill: frame == 51}
		if health <= 0 {
			row.ObserverRespawn = true
			row.Command.Buttons = 1
		}
		rows = append(rows, row)
	}
	return rows
}

func TestObserverCycleRequiresDeathRespawnAndRecovery(t *testing.T) {
	f := ObserverRespawn{AfterFrames: 1, TimeoutFrames: 6, RecoveryFrames: 3}
	if r := checkObserverCycle(f, 50, cycleFixture()); !r.Passed || r.DeathFrame != 52 || r.RespawnFrame != 55 || r.RecoveryFrames != 3 {
		t.Fatal(r)
	}
	for _, mutate := range []func([]Trace) []Trace{
		func(r []Trace) []Trace { return r[:7] },
		func(r []Trace) []Trace { r[1].ObserverKill = false; return r },
		func(r []Trace) []Trace { r[2].Health = nil; return r },
		func(r []Trace) []Trace { r[3].Command.Forward = 100; return r },
		func(r []Trace) []Trace { r[3].ObserverRespawn = false; return r },
		func(r []Trace) []Trace { r[7].ObserverRespawn = true; return r },
		func(r []Trace) []Trace {
			for i := 2; i < len(r); i++ {
				health := int16(0)
				r[i].Health = &health
			}
			return r
		},
	} {
		if r := checkObserverCycle(f, 50, mutate(cycleFixture())); r.Passed {
			t.Fatal("incomplete/invalid cycle accepted")
		}
	}
}

func TestMotionExcludesDeathAndRespawnJump(t *testing.T) {
	alive, dead := int16(100), int16(0)
	rows := []Trace{{Self: &quake.Vec3{}, Health: &alive}, {Self: &quake.Vec3{100, 0, 0}, Health: &dead}, {Self: &quake.Vec3{1000, 0, 0}, Health: &alive}}
	r := measureMotion(fixture(), rows, false)
	if r.PathUnits != 0 || r.ExcludedDeadIntervals != 2 {
		t.Fatal(r)
	}
}
