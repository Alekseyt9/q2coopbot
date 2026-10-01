package quake

import "testing"

func TestGrenadeObservationUsesOnlyCurrentEntity(t *testing.T) {
	d := Decoder{Config: map[int]string{33: "models/objects/grenade2/tris.md2", 34: "models/objects/grenade/tris.md2"}}
	f := Frame{Number: 1, Entities: map[int]Entity{8: {Number: 8, Model: 1, Origin: Vec3{10, 20, 30}}, 9: {Number: 9, Model: 2}}}
	s := d.Snapshot(f)
	if len(s.Projectiles) != 1 || s.Projectiles[0].ID != 8 || s.Projectiles[0].Origin != (Vec3{10, 20, 30}) {
		t.Fatal(s.Projectiles)
	}
	if s = d.Snapshot(Frame{Number: 2}); len(s.Projectiles) != 0 {
		t.Fatal("Expired observation retained")
	}
	f.Entities[8] = Entity{Number: 8, Model: 2}
	if s = d.Snapshot(f); len(s.Projectiles) != 0 {
		t.Fatal("Reused ID classified as grenade")
	}
}
