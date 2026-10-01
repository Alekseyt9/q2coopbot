package quake

import (
	"math"
	"testing"
)

func TestProjectileBoxFindsInteriorObstacle(t *testing.T) {
	planes, brush := testBoxBrush(Vec3{-1, -1, -1}, Vec3{1, 1, 1}, 0)
	c := &CollisionMap{planes: planes, brushes: []bspBrush{brush}, worldBrushes: []int{0}, sides: []uint16{0, 1, 2, 3, 4, 5}}
	for _, x := range []float64{-10, 10} {
		for _, y := range []float64{-10, 10} {
			for _, z := range []float64{-10, 10} {
				p := Vec3{x, y, z}
				if c.traceProjectile(p, p).StartSolid {
					t.Fatal("Corner inside obstacle")
				}
			}
		}
	}
	if clear, valid := c.projectileBoxClear(Vec3{-10, -10, -10}, Vec3{10, 10, 10}); clear || !valid {
		t.Fatal("Corner sampling missed interior obstacle", clear, valid)
	}
	if clear, valid := c.projectileBoxClear(Vec3{2, -10, -10}, Vec3{10, 10, 10}); !clear || !valid {
		t.Fatal("Separated box rejected", clear, valid)
	}
	if clear, _ := c.projectileBoxClear(Vec3{1.01, -1, -1}, Vec3{2, 1, 1}); clear {
		t.Fatal("Trace epsilon ignored")
	}
	c.brushes[0].contents = 32 // Water alone is not MASK_SHOT.
	if clear, valid := c.projectileBoxClear(Vec3{-10, -10, -10}, Vec3{10, 10, 10}); !clear || !valid {
		t.Fatal("Wrong collision mask")
	}
	if _, valid := c.projectileBoxClear(Vec3{math.NaN(), 0, 0}, Vec3{1, 1, 1}); valid {
		t.Fatal("Invalid coordinates accepted")
	}
	if _, valid := c.projectileBoxClear(Vec3{2, 0, 0}, Vec3{1, 1, 1}); valid {
		t.Fatal("Inverted bounds accepted")
	}
	if _, valid := (*MapInfo)(nil).ProjectileBoxClear(Vec3{}, Vec3{}); valid {
		t.Fatal("Missing geometry accepted")
	}
}
