package quake

import "testing"

func TestAimPointBounds(t *testing.T) {
	for _, tc := range []struct {
		name  string
		solid uint16
		z     float64
	}{
		{"standing", 2 | 3<<5 | 8<<10, 22},
		{"ducking", 2 | 3<<5 | 4<<10, -8},
		{"short", 2 | 1<<5 | 4<<10, -4},
		{"missing", 0, 22}, {"brush", 31, 22},
		{"invalid", 2, 22},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := Object{Origin: Vec3{10, 20, 30}, Solid: tc.solid}
			if got := o.AimPoint(); got != (Vec3{10, 20, 30 + tc.z}) {
				t.Fatalf("aim=%v", got)
			}
		})
	}
}
