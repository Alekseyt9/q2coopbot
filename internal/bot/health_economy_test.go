package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestHealthEconomy(t *testing.T) {
	for _, tc := range []struct {
		hp     int16
		amount int
		want   bool
	}{
		{99, 25, false}, {90, 25, false}, {76, 25, false}, {75, 25, true},
		{91, 10, false}, {90, 10, true}, {99, 2, true}, {99, 100, true},
		{100, 25, false}, {90, 0, false}, {44, 0, true},
	} {
		if got := usefulHealth(quake.Snapshot{Health: tc.hp}, quake.Object{HealthAmount: tc.amount}); got != tc.want {
			t.Errorf("hp=%d amount=%d: got %v", tc.hp, tc.amount, got)
		}
	}
}

func TestRecoveryRequestPreservesOversizedKit(t *testing.T) {
	kit := quake.Object{Class: "item_health", Origin: quake.Vec3{20, 0, 15}, HealthAmount: 25}
	p := &Planner{healthActive: true, healthTarget: healthStand(kit.Origin)}
	s := quake.Snapshot{Frame: 50, Health: 95, Pickups: []quake.Object{kit}}
	if _, ok := p.healthGoal(s); ok {
		t.Fatal("active recovery wastes a kit after health restored")
	}
	s.Health = 40
	if _, ok := p.healthGoal(s); !ok {
		t.Fatal("critical health must permit recovery")
	}
}
