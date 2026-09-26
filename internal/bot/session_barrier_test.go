package bot

import "testing"

func TestBarrierNeedsTwoMatchingRoles(t *testing.T) {
	a := sessionReady{Map: "base2", Generation: 7, Phase: 1, Frame: 32, Role: "actor"}
	b := a
	b.Frame = 77
	b.Role = "observer"
	if start, err := sessionBarrierStart(a, b, "base2", 7, 1); err != nil || start != 87 {
		t.Fatal(start, err)
	}
	for _, change := range []func(*sessionReady){func(r *sessionReady) { r.Role = "actor" }, func(r *sessionReady) { r.Generation = 8 }, func(r *sessionReady) { r.Map = "base1" }, func(r *sessionReady) { r.Phase = 0 }, func(r *sessionReady) { r.Frame = 0 }} {
		bad := b
		change(&bad)
		if _, err := sessionBarrierStart(a, bad, "base2", 7, 1); err == nil {
			t.Fatal("invalid readiness accepted")
		}
	}
}
