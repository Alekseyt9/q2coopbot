package quake

import "testing"

func TestRailTrailPayloadAndTruncation(t *testing.T) {
	packet := append([]byte{3, 3}, make([]byte, 12)...)
	packet = append(packet, 11, 'o', 'k', 0)
	d := NewDecoder()
	if _, err := d.Parse(packet); err != nil || len(d.Commands) != 1 || d.Commands[0] != "ok" {
		t.Fatalf("rail trail: %v %+v", err, d.Commands)
	}
	for n := 0; n < 12; n++ {
		d = NewDecoder()
		if _, err := d.Parse(packet[:2+n]); err == nil {
			t.Fatalf("truncated payload%d accepted", n)
		}
	}
}
