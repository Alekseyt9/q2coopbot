package bot

import (
	"q2coopbot/internal/quake"
	"testing"
)

func TestNoCoverageRequiresEvidenceAcrossAllSafeCandidates(t *testing.T) {
	if !(&SearchVisibility{HiddenSamples: 5, SafeCandidates: 3, AuditComplete: true, AuditSamples: 20}).noNewCoverage() {
		t.Fatal("zero-gain candidates not rejected")
	}
	for _, v := range []*SearchVisibility{nil, {}, {HiddenSamples: 5}, {SafeCandidates: 3},
		{HiddenSamples: 5, SafeCandidates: 3},
		{HiddenSamples: 5, SafeCandidates: 3, AuditSamples: 128},
		{HiddenSamples: 5, SafeCandidates: 3, AuditComplete: true, AuditSamples: 20, AuditMaxGain: 1},
		{HiddenSamples: 5, SafeCandidates: 3, MaxNewlyVisible: 1}} {
		if v.noNewCoverage() {
			t.Fatalf("missing evidence or useful alternative rejected: %+v", v)
		}
	}
}

func TestVisibilityPreferenceIsBounded(t *testing.T) {
	near := searchViewpointScore(100, 100, 0)
	if searchViewpointScore(140, 100, 4) >= near {
		t.Fatal("new coverage did not justify a short detour")
	}
	if searchViewpointScore(240, 100, 32) <= near {
		t.Fatal("coverage rewarded an excessive detour")
	}
	if searchViewpointScore(120, 100, 0) <= near {
		t.Fatal("zero gain changed distance ordering")
	}
}

func TestVisibilityUsesEyeHeightAndOcclusion(t *testing.T) {
	samples := []quake.Vec3{{100, 0, 24}, {200, 0, 24}}
	gain := newVisibleSamples(samples, quake.Vec3{0, 0, 24}, func(from, to quake.Vec3) bool {
		if from[2] != 46 || to[2] != 46 {
			t.Fatal("rays are not at eye height")
		}
		return to[0] < 150
	})
	if gain != 1 {
		t.Fatalf("occluded sample counted: %d", gain)
	}
}

func TestAlreadyVisibleSamplesGiveNoNovelCoverage(t *testing.T) {
	current, previous, point := quake.Vec3{1, 0, 24}, quake.Vec3{2, 0, 24}, quake.Vec3{3, 0, 24}
	for _, visibleFrom := range []float64{1, 2} {
		clear := func(from, to quake.Vec3) bool { return from[0] == visibleFrom }
		if unseenSearchSample(point, current, previous, true, clear) {
			t.Fatal("previously visible sample rewarded as new")
		}
	}
	if !unseenSearchSample(point, current, previous, false, func(from, to quake.Vec3) bool { return from[0] == 2 }) {
		t.Fatal("unknown previous position excluded a sample")
	}
}
