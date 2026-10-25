package quake

import "testing"

// Stock Yamagi CL_ParseTEnt reads two positions for TE_BFG_LASER and
// one position for each of TE_BFG_EXPLOSION/TE_BFG_BIGEXPLOSION.
func TestBFGEffectsKeepFollowingMessageAligned(t *testing.T) {
	for _, tc := range []struct{ kind, size byte }{{23, 12}, {20, 6}, {21, 6}} {
		packet := append([]byte{3, tc.kind}, make([]byte, tc.size)...)
		packet = append(packet, 11, 'o', 'k', 0)
		d := NewDecoder()
		if _, err := d.Parse(packet); err != nil {
			t.Fatalf("effect %d: %v", tc.kind, err)
		}
		if len(d.Commands) != 1 || d.Commands[0] != "ok" {
			t.Fatalf("effect %d swallowed next message: %v", tc.kind, d.Commands)
		}
		if tc.kind != 23 && (len(d.Explosions) != 1 || d.Explosions[0].Kind != tc.kind) {
			t.Fatal("BFG explosion missing", d.Explosions)
		}
		for n := 2; n < 2+int(tc.size); n++ {
			if _, err := NewDecoder().Parse(packet[:n]); err == nil {
				t.Fatalf("effect %d accepted truncated payload of %d bytes", tc.kind, n-2)
			}
		}
	}
}
