package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestInventoryWatchDoesNotToggleOpenInventory(t *testing.T) {
	w := inventoryWatch{}
	s := quake.Snapshot{Map: "base1", Frame: 1, Health: 100}
	if w.command(s) != "" {
		t.Fatal("inventory requested during initial config burst")
	}
	s.Frame = 4
	if w.command(s) != "inven" {
		t.Fatal("no request after warmup")
	}
	for frame := 5; frame <= 100; frame++ {
		s.Frame = frame
		s.InventoryOpen = true
		if w.command(s) != "" {
			t.Fatal("toggled open inventory")
		}
	}
	s.InventoryOpen = false
	s.Frame = 101
	if w.command(s) != "inven" {
		t.Fatal("closed inventory not reopened")
	}
	s.Frame = 102
	if w.command(s) != "" {
		t.Fatal("request spam")
	}
	s.Health = 0
	s.Frame = 103
	if w.command(s) != "" {
		t.Fatal("dead request")
	}
	s.Health = 100
	s.Frame = 104
	w.command(s)
	s.Frame = 107
	if w.command(s) != "inven" {
		t.Fatal("respawn did not reset request")
	}
}

func TestWeaponSwitchUsesInventoryAndBoundsRetries(t *testing.T) {
	w := weaponSwitch{}
	s := quake.Snapshot{Map: "base1", Health: 100, Weapon: "models/weapons/v_shotg/tris.md2", InventoryKnown: true,
		Inventory: []quake.InventoryItem{{Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 20}}}
	for frame := 1; frame <= 60; frame++ {
		s.Frame = frame
		got := w.command(s)
		want := ""
		if frame == 3 {
			want = "use Machinegun"
		}
		if frame == 23 || frame == 43 {
			want = "use Blaster"
		}
		if got != want {
			t.Fatalf("frame %d: %q want %q", frame, got, want)
		}
	}
	s.Ammo = 10
	s.Frame = 61
	w.command(s)
	s.Ammo = 0
	s.InventoryAgeFrames = 21
	s.Frame = 62
	w.command(s)
	s.Frame = 64
	if got := w.command(s); got != "weapnext" {
		t.Fatalf("stale inventory fallback: %q", got)
	}
	s.InventoryAgeFrames = 0
	s.Inventory = []quake.InventoryItem{{Name: "Super Shotgun", Count: 1}, {Name: "Shells", Count: 1}, {Name: "BFG10K", Count: 1}, {Name: "Cells", Count: 49}}
	if got := stockedWeapon(s); got != "use Blaster" {
		t.Fatalf("insufficient ammo selected: %q", got)
	}
	s.Health = 0
	s.Frame = 65
	if w.command(s) != "" {
		t.Fatal("dead switch")
	}
}
