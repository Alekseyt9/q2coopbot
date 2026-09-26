package quake

import "testing"

func TestSnapshotHealthAmounts(t *testing.T) {
	for path, amount := range map[string]int{
		"models/items/healing/stimpack/tris.md2": 2,
		"models/items/healing/medium/tris.md2":   10,
		"models/items/healing/large/tris.md2":    25,
		"models/items/mega_h/tris.md2":           100,
	} {
		d := NewDecoder()
		d.Config[33] = path
		s := d.Snapshot(Frame{Entities: map[int]Entity{10: {Number: 10, Model: 1}}})
		if len(s.Pickups) != 1 || s.Pickups[0].Class != "item_health" || s.Pickups[0].HealthAmount != amount {
			t.Fatalf("%s: %+v", path, s.Pickups)
		}
	}
}

func TestSnapshotRecognizesWorldPickupsButNotViewWeapons(t *testing.T) {
	d := NewDecoder()
	paths := []string{"models/weapons/g_machn/tris.md2", "models/items/ammo/bullets/medium/tris.md2", "models/items/armor/jacket/tris.md2", "models/weapons/v_machn/tris.md2"}
	f := Frame{Entities: map[int]Entity{}}
	for i, path := range paths {
		d.Config[33+i] = path
		f.Entities[10+i] = Entity{Number: 10 + i, Model: 1 + i}
	}
	s := d.Snapshot(f)
	if len(s.Pickups) != 3 {
		t.Fatalf("pickups=%+v", s.Pickups)
	}
	found := map[string]bool{}
	for _, item := range s.Pickups {
		found[item.Class] = true
	}
	for i, want := range []string{"weapon_machinegun", "ammo_bullets", "item_armor_jacket"} {
		if !found[want] {
			t.Fatalf("%d: %s", i, s.Pickups[i].Class)
		}
	}
}
