package bot

import (
	"os"
	"path/filepath"
	"testing"

	"q2coopbot/internal/quake"
)

func TestBase1YieldThenReleaseAndInterrupt(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("requires local base1 BSP/AAS")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps/base1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	team := quake.Vec3{-344, -544, -79.875}
	item := quake.Object{ID: 42, Class: "item_armor_jacket", Origin: quake.Vec3{-384, -544, -88.875}}
	s := quake.Snapshot{Frame: 40, Health: 100, OnGround: true, InventoryKnown: true, Self: quake.Vec3{-300, -600, -79.875}, Teammate: &team, Pickups: []quake.Object{item}}
	p := &Planner{Nav: n, World: World{Map: "base1", Geometry: &g, Goal: "follow_teammate"}}
	if _, ok := p.pickupGoal(s); ok || p.World.ResourceYield == nil {
		t.Fatal("did not yield reachable armor")
	}
	team = quake.Vec3{-300, -700, -79.875}
	s.Frame++
	if _, ok := p.pickupGoal(s); !ok || p.World.ResourceYield != nil {
		t.Fatal("armor remains reserved after teammate leaves")
	}
	team = quake.Vec3{-344, -544, -79.875}
	s.Frame++
	if _, ok := p.pickupGoal(s); ok || p.pickup != nil || p.World.Pickup.State != "yielded" {
		t.Fatal("active approach not cancelled")
	}
	s.Frame++
	if _, ok := p.pickupGoal(s); ok {
		t.Fatal("cancelled approach immediately restarted")
	}
}

func TestTeammatePickupPriority(t *testing.T) {
	at := quake.Vec3{0, 0, 0}
	for _, tc := range []struct {
		name, class   string
		self, team    quake.Vec3
		visible, want bool
	}{
		{"closer armor", "item_armor_jacket", quake.Vec3{150, 0, 0}, quake.Vec3{60, 0, 0}, true, true},
		{"closer ammo", "ammo_shells", quake.Vec3{150, 0, 0}, quake.Vec3{60, 0, 0}, true, true},
		{"coop weapon", "weapon_shotgun", quake.Vec3{150, 0, 0}, quake.Vec3{60, 0, 0}, true, false},
		{"critical health unaffected", "item_health", quake.Vec3{150, 0, 0}, quake.Vec3{60, 0, 0}, true, false},
		{"similar distance", "ammo_shells", quake.Vec3{80, 0, 0}, quake.Vec3{60, 0, 0}, true, false},
		{"bot closer", "ammo_shells", quake.Vec3{30, 0, 0}, quake.Vec3{60, 0, 0}, true, false},
		{"different floor", "ammo_shells", quake.Vec3{150, 0, 0}, quake.Vec3{0, 0, 64}, true, false},
		{"distant teammate", "ammo_shells", quake.Vec3{250, 0, 0}, quake.Vec3{120, 0, 0}, true, false},
		{"lost teammate", "ammo_shells", quake.Vec3{150, 0, 0}, quake.Vec3{60, 0, 0}, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := quake.Snapshot{Self: tc.self}
			if tc.visible {
				s.Teammate = &tc.team
			}
			if got := teammatePickupPriority(s, tc.class, at); got != tc.want {
				t.Fatalf("got %v want %v", got, tc.want)
			}
		})
	}
}
