package quake

import "testing"

func TestBarrelObservationCurrentSolidOnly(t *testing.T) {
	d := Decoder{Config: map[int]string{33: "models/objects/barrels/tris.md2"}}
	f := Frame{Number: 1, Entities: map[int]Entity{8: {Number: 8, Model: 1, Solid: 2, Origin: Vec3{10, 20, 24}}}}
	if got := d.Snapshot(f).Barrels; len(got) != 1 || got[0].ID != 8 || got[0].Origin != f.Entities[8].Origin {
		t.Fatal(got)
	}
	f.Entities[8] = Entity{Number: 8, Model: 1}
	if len(d.Snapshot(f).Barrels) != 0 || len(d.Snapshot(Frame{Number: 2}).Barrels) != 0 {
		t.Fatal("nonsolid or expired barrel retained")
	}
}
