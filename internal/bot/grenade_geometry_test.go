package bot

import (
	"os"
	"q2coopbot/internal/quake"
	"testing"
)

func TestGrenadeGeometryPrefixOnBase1(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("Q2_SEARCH_SCAN_ROOT not set")
	}
	g, err := quake.LoadMap(root, "base1")
	if err != nil {
		t.Fatal(err)
	}
	s := quake.Snapshot{Gravity: 800}
	start := quake.Vec3{32, -344, 38.125}
	r := grenadeGeometryEnvelope(s, start, quake.Vec3{0, 1, 0}, quake.Vec3{1, 0, 0}, quake.Vec3{0, 0, 1}, &g)
	if r.Reason != "free_prefix_only" || r.ClearSeconds != .5 || r.Authorized || r.PostBounceCertified {
		t.Fatal(r)
	}
	r = grenadeGeometryEnvelope(s, start, quake.Vec3{0, 0, -1}, quake.Vec3{0, -1, 0}, quake.Vec3{1, 0, 0}, &g)
	if r.Reason != "possible_static_bounce" || r.ClearSeconds != 0 || r.StopSeconds != .1 || r.Authorized || r.PostBounceCertified {
		t.Fatal(r)
	}
	s.Movers = []quake.Mover{{Model: 999999}}
	r = grenadeGeometryEnvelope(s, start, quake.Vec3{0, 1, 0}, quake.Vec3{1, 0, 0}, quake.Vec3{0, 0, 1}, &g)
	if r.Reason != "unknown_mover_bounds" || r.ClearSeconds != 0 {
		t.Fatal(r)
	}
	s.Movers = nil
	r = grenadeGeometryEnvelope(s, quake.Vec3{136, -224, 38.125}, quake.Vec3{1, 0, 0}, quake.Vec3{0, -1, 0}, quake.Vec3{0, 0, 1}, &g)
	if r.Reason != "possible_static_bounce" || r.ClearSeconds != .2 || r.StopSeconds < .299 || r.StopSeconds > .301 {
		t.Fatal(r)
	}
	if r.SurfaceContact != nil {
		t.Fatal("grazing family incorrectly certified")
	}
	r = grenadeGeometryEnvelope(s, quake.Vec3{140, -224, 38.125}, quake.Vec3{1, 0, 0}, quake.Vec3{0, -1, 0}, quake.Vec3{0, 0, 1}, &g)
	if r.SurfaceContact == nil || r.SurfaceBounceEnvelope == nil {
		t.Fatalf("common wall missing: %+v", r)
	}
	if r.SurfaceContact.Max[0]-r.SurfaceContact.Min[0] > 0.001 || r.SurfaceBounceEnvelope.Authorized || r.SurfaceBounceEnvelope.GeometryCertified {
		t.Fatal(r.SurfaceContact)
	}
	model, ok := g.Model(1)
	if !ok {
		t.Fatal("Native inline model missing")
	}
	mover := quake.Mover{Model: 1}
	for i := range mover.Origin {
		mover.Origin[i] = start[i] - (model.Min[i]+model.Max[i])/2
	}
	s.Movers = []quake.Mover{mover}
	r = grenadeGeometryEnvelope(s, start, quake.Vec3{0, 1, 0}, quake.Vec3{1, 0, 0}, quake.Vec3{0, 0, 1}, &g)
	if r.Reason != "observed_mover_overlap" || r.ClearSeconds != 0 || r.PostBounceCertified {
		t.Fatal(r)
	}
}

func TestGrenadeGeometryMissingMapNeverCertifies(t *testing.T) {
	r := grenadeGeometryEnvelope(quake.Snapshot{Gravity: 800}, quake.Vec3{}, quake.Vec3{1, 0, 0}, quake.Vec3{0, -1, 0}, quake.Vec3{0, 0, 1}, nil)
	if r.Reason != "unknown_geometry" || r.ClearSeconds != 0 || r.Authorized || r.PostBounceCertified {
		t.Fatal(r)
	}
}
