package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestHealthObservationMaskBoundaries(t *testing.T) {
	c := Client{testHideHealthFrames: []int{10, 30}, testTeleportSent: true, testTeleportMap: "base1", testScenarioFrameOrigin: 40}
	for _, frame := range []int{49, 50, 69, 70} {
		s := quake.Snapshot{Map: "base1", Frame: frame, Health: 40, Pickups: []quake.Object{{Class: "item_health"}, {Class: "item_armor_jacket"}}}
		got := c.maskTestHealth(s)
		want := 2
		if frame >= 50 && frame < 70 {
			want = 1
		}
		if len(got.Pickups) != want || got.Health != 40 || len(s.Pickups) != 2 {
			t.Fatalf("frame%d: %+v", frame, got)
		}
	}
	for _, ready := range []bool{false, true} {
		c.testTeleportSent = ready
		s := quake.Snapshot{Map: "base2", Frame: 55, Pickups: []quake.Object{{Class: "item_health"}}}
		if c.testHealthMasked(s) {
			t.Fatal("mask applied outside fixture map")
		}
		s.Map = "base1"
		if !ready && c.testHealthMasked(s) {
			t.Fatal("mask applied before setup")
		}
	}
}
