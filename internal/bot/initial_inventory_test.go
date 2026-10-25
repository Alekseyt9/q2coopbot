package bot

import (
	"q2coopbot/internal/quake"
	"strings"
	"testing"
)

func exactInventoryFixture() []quake.InventoryItem {
	var items []quake.InventoryItem
	for _, name := range initialInventoryNames() {
		n := 0
		if name == "Blaster" || name == "Railgun" || name == "BFG10K" {
			n = 1
		}
		if name == "Slugs" {
			n = 3
		}
		if name == "Cells" {
			n = 49
		}
		items = append(items, quake.InventoryItem{Name: name, Count: n})
	}
	return items
}

func TestExactInitialInventoryFreshnessAndMissingStock(t *testing.T) {
	items := exactInventoryFixture()
	command, err := initialInventoryCommand(items, "Railgun")
	if err != nil || !strings.HasPrefix(command, "test_inventory_v1 ") {
		t.Fatal(command, err)
	}
	if _, err := initialInventoryCommand(items, "BFG10K"); err == nil {
		t.Fatal("undercharged BFG equipped")
	}
	if _, err := initialInventoryCommand(items, "Shotgun;quit"); err == nil {
		t.Fatal("unsafe item accepted")
	}
	s := quake.Snapshot{InventoryKnown: true, InventoryAgeFrames: 0, Inventory: items, Weapon: "models/weapons/v_rail/tris.md2"}
	if !initialInventoryReady(items, "Railgun", s) {
		t.Fatal("fresh exact inventory rejected")
	}
	s.InventoryAgeFrames = 3
	if initialInventoryReady(items, "Railgun", s) {
		t.Fatal("stale inventory released")
	}
	s.InventoryAgeFrames = 0
	s.Inventory = append([]quake.InventoryItem(nil), items...)
	for i := range s.Inventory {
		if s.Inventory[i].Name == "Shotgun" {
			s.Inventory[i].Count = 1
		}
	}
	if initialInventoryReady(items, "Railgun", s) {
		t.Fatal("unexpected owned shotgun released")
	}
	if _, err := initialInventoryCommand(append(items[:15:15], items[0]), "Railgun"); err == nil {
		t.Fatal("duplicate inventory item accepted")
	}
}
