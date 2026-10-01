package quake

import "testing"

func TestExplosionPackedPositionAndPacketLifetime(t *testing.T) {
	d := NewDecoder()
	packet := []byte{3, 7, 8, 0, 240, 255, 24, 0, 6}
	if _, err := d.Parse(packet); err != nil {
		t.Fatal(err)
	}
	if len(d.Explosions) != 1 || d.Explosions[0].Kind != 7 || d.Explosions[0].Position != (Vec3{1, -2, 3}) {
		t.Fatal(d.Explosions)
	}
	if _, err := d.Parse([]byte{6}); err != nil || len(d.Explosions) != 0 {
		t.Fatal("Explosion persisted across packets", err)
	}
	for n := 2; n < 8; n++ {
		if _, err := d.Parse(packet[:n]); err == nil || len(d.Explosions) != 0 {
			t.Fatalf("partial event published at%d", n)
		}
	}
}
