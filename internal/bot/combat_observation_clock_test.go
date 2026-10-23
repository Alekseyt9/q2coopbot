package bot

import "testing"

// Scalar clock contract only; no neural network or game client is run.
func TestSynchronousCombatObservationAge(t *testing.T) {
	for _, tc := range []struct {
		wall           int64
		sync, released bool
		want           int64
	}{
		{9, false, true, 9}, {9, true, false, 9}, {0, true, true, 0}, {9, true, true, 0},
		{300, true, true, 0}, {301, true, true, 301}, {-1, true, true, -1},
	} {
		if got := combatObservationAge(tc.wall, tc.sync, tc.released); got != tc.want {
			t.Fatalf("wall=%d sync=%v released=%v: got %d want %d", tc.wall, tc.sync, tc.released, got, tc.want)
		}
	}
}
