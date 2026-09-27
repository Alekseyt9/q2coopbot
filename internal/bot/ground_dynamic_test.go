package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestGroundDynamicBounds(t *testing.T) {
	g := &quake.MapInfo{Models: []quake.BSPModel{{}, {Min: quake.Vec3{100, 100, 0}, Max: quake.Vec3{120, 120, 64}}}}
	for _, tc := range []struct {
		name   string
		self   quake.Vec3
		movers []quake.Mover
		want   int
	}{
		{"spawn", quake.Vec3{80, 100, 24}, nil, 1},
		{"observed_translation", quake.Vec3{0, 0, 24}, []quake.Mover{{Model: 1, Origin: quake.Vec3{-100, -100, 0}}}, 1},
		{"moved_away_spawn_still_excluded", quake.Vec3{100, 100, 24}, []quake.Mover{{Model: 1, Origin: quake.Vec3{500, 500, 0}}}, 1},
		{"distant", quake.Vec3{0, 0, 24}, nil, 0},
		{"vertical_clearance", quake.Vec3{100, 100, 120}, nil, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := nearbyGroundBrushModel(quake.Snapshot{Self: tc.self, Movers: tc.movers}, g); got != tc.want {
				t.Fatalf("got %d want %d", got, tc.want)
			}
		})
	}
}
