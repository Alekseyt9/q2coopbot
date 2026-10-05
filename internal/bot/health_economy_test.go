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

func TestStimpackRecoveryDuringCombat(t *testing.T) {
	visible, hidden := true, false
	s := quake.Snapshot{Health: 10, Self: quake.Vec3{0, 0, 24}, Enemies: []quake.Object{{Origin: quake.Vec3{200, 0, 24}, ClearShot: &visible}}}
	item := quake.Object{Class: "item_health", Origin: quake.Vec3{200, 0, 14.875}, HealthAmount: 2}
	if healthRecoveryUseful(s, item) {
		t.Fatal("distant two-HP recovery diverted a critical-health fight")
	}
	item.Origin[0] = 96
	if !healthRecoveryUseful(s, item) {
		t.Fatal("nearby stimpack was rejected")
	}
	item.Origin[0] = 200
	for _, amount := range []int{0, 10, 25, 100} {
		item.HealthAmount = amount
		if !healthRecoveryUseful(s, item) {
			t.Fatalf("rejected meaningful or unknown kit: %d", amount)
		}
	}
	item.HealthAmount = 2
	s.Enemies[0].ClearShot = &hidden
	if !healthRecoveryUseful(s, item) {
		t.Fatal("occluded enemy prevented ordinary recovery")
	}
	s.Enemies[0].ClearShot = &visible
	s.Enemies[0].Origin[0] = 651
	if !healthRecoveryUseful(s, item) {
		t.Fatal("distant enemy prevented ordinary recovery")
	}
}

func TestActiveStimpackRecoveryReconsidersVisibleThreat(t *testing.T) {
	visible := true
	stim := quake.Object{Class: "item_health", Origin: quake.Vec3{200, 0, 14.875}, HealthAmount: 2}
	kit := quake.Object{Class: "item_health", Origin: quake.Vec3{240, 0, 14.875}, HealthAmount: 25}
	p := &Planner{healthActive: true, healthTarget: healthStand(stim.Origin)}
	s := quake.Snapshot{Health: 10, Self: quake.Vec3{0, 0, 24}, Pickups: []quake.Object{stim, kit}, Enemies: []quake.Object{{Origin: quake.Vec3{180, 0, 24}, ClearShot: &visible}}}
	if at, ok := p.healthGoal(s); !ok || at != healthStand(kit.Origin) {
		t.Fatal("active stimpack held recovery instead of a meaningful kit", at, ok)
	}
	s.Pickups = []quake.Object{stim}
	if _, ok := p.healthGoal(s); ok {
		t.Fatal("stimpack reacquired during the same visible fight")
	}
}
