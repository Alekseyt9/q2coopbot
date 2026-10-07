package quake

import "testing"

func TestBubbleTrailConsumesBothPositions(t *testing.T) {
	payload := append([]byte{3, 11}, make([]byte, 12)...)
	payload = append(payload, 11, 1, 'o', 'k', 0) // svc_print after the effect
	d := NewDecoder()
	if _, err := d.Parse(payload); err != nil {
		t.Fatal(err)
	}
	if _, err := NewDecoder().Parse(payload[:13]); err == nil {
		t.Fatal("truncated bubble trail accepted")
	}
}
