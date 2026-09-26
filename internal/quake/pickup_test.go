package quake

import "testing"

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
