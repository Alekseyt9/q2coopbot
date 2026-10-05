package quake

import "testing"

func TestGrenadeObservationUsesOnlyCurrentEntity(t *testing.T) {
	d := Decoder{Config: map[int]string{33: "models/objects/grenade2/tris.md2", 34: "models/objects/grenade/tris.md2"}}
	f := Frame{Number: 1, Entities: map[int]Entity{8: {Number: 8, Model: 1, Origin: Vec3{10, 20, 30}}, 9: {Number: 9, Model: 2}}}
	s := d.Snapshot(f)
	if len(s.Projectiles) != 2 {
		t.Fatal(s.Projectiles)
	}
	seen := map[int]Object{}
	for _, o := range s.Projectiles {
		seen[o.ID] = o
	}
	if seen[8].Class != "hand_grenade" || seen[8].Origin != (Vec3{10, 20, 30}) || seen[9].Class != "grenade" {
		t.Fatal(seen)
	}
	if s = d.Snapshot(Frame{Number: 2}); len(s.Projectiles) != 0 {
		t.Fatal("Expired observation retained")
	}
	f.Entities = map[int]Entity{8: {Number: 8, Model: 3}}
	if s = d.Snapshot(f); len(s.Projectiles) != 0 {
		t.Fatal("Reused ID classified as grenade")
	}
}
