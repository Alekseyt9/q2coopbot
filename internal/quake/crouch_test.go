package quake

import "testing"

func TestCrouchPassageGeometry(t *testing.T) {
	for _, tc := range []struct {
		name              string
		ceiling, floorEnd float64
		want              bool
	}{
		{"passage", 40, 200, true}, {"too_low", 26, 200, false}, {"edge", 40, 8, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &CollisionMap{}
			add := func(lo, hi Vec3) {
				planes, brush := testBoxBrush(lo, hi, len(c.sides))
				for range planes {
					c.sides = append(c.sides, uint16(len(c.sides)))
				}
				c.planes = append(c.planes, planes...)
				c.worldBrushes = append(c.worldBrushes, len(c.brushes))
				c.brushes = append(c.brushes, brush)
			}
			add(Vec3{-200, -100, -32}, Vec3{tc.floorEnd, 100, 0})
			add(Vec3{20, -100, tc.ceiling}, Vec3{80, 100, 100})
			g := &MapInfo{collision: c}
			p := Vec3{0, 0, 24.125}
			if g.PlayerMoveClear(p, Vec3{16, 0, 24.125}) {
				t.Fatal("standing fits under low roof")
			}
			if got := g.CrouchStepClear(p, 1, 0, 16); got != tc.want {
				t.Fatalf("crouch=%v want=%v", got, tc.want)
			}
			if !g.PlayerMoveClear(Vec3{-32, 0, 24.125}, Vec3{-32, 0, 24.125}) {
				t.Fatal("entry blocked")
			}
			if g.PlayerMoveClear(Vec3{48, 0, 24.125}, Vec3{48, 0, 24.125}) {
				t.Fatal("standing inside roof allowed")
			}
			if !g.PlayerMoveClear(Vec3{112, 0, 24.125}, Vec3{112, 0, 24.125}) {
				t.Fatal("cannot stand after exit")
			}
		})
	}
	var missing *MapInfo
	if missing.CrouchStepClear(Vec3{}, 1, 0, 16) {
		t.Fatal("missing geometry accepted")
	}
}

func TestSnapshotDuckFlagAndEye(t *testing.T) {
	d := NewDecoder()
	for _, flags := range []byte{4, 5, 4} {
		s := d.Snapshot(Frame{PMFlags: flags, Origin: Vec3{0, 0, 24}})
		want := 46.0
		if flags == 5 {
			want = 22
		}
		if !s.OnGround || s.Ducked != (flags == 5) || s.EyePoint()[2] != want {
			t.Fatalf("flags%d: %+v", flags, s)
		}
	}
}
