package policy

import (
	"encoding/json"
	"math"
	"testing"
)

// Metadata, PRNG and pure transforms only. No Go neural evaluation.
func TestJointAimPureContract(t *testing.T) {
	for frame := 1; frame <= 20000; frame++ {
		g, u := JointAimUniforms(20261023, frame, 1)
		if g < 0 || g >= 1 || u < 0 || u >= 1 {
			t.Fatal("uniform outside domain")
		}
		k := 0
		if g >= .5 {
			k = 1 + int(u*24)
		}
		if k != GeometryAimExploreCandidate(20261023, frame, 1) {
			t.Fatal("initial sampler differs", frame)
		}
	}
	raw := [4]float64{.1, -.2, 20, -20}
	for mode := 0; mode <= 1; mode++ {
		for k := 0; k < 25; k++ {
			z, err := ApplyJointAimOffsets(raw, mode, k)
			if err != nil || z[0] != raw[0] || z[1] != raw[1] {
				t.Fatal("source motion changed", err)
			}
			if k == 0 && z != raw {
				t.Fatal("baseline latent changed")
			}
			for _, v := range z {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatal("nonfinite transform")
				}
			}
		}
	}
	for _, bad := range []int{-1, 25} {
		if _, err := ApplyJointAimOffsets(raw, 0, bad); err == nil {
			t.Fatal("invalid category accepted")
		}
	}
	f, c, h := make([]float64, 1147), make([]float64, 185), make([]float64, 25)
	x, err := JointAimCandidateInputs(f, c, h, [2]float64{.25, 1})
	if err != nil || len(x) != 1359 || x[1357] != .25 || x[1358] != 1 {
		t.Fatal("packing changed", err)
	}
	c[77], c[99], c[147] = 1, 1, 1
	if !jointAimActive(c) {
		t.Fatal("MG target not eligible")
	}
	c[151] = 1
	if jointAimActive(c) {
		t.Fatal("none target eligible")
	}
	f[0] = math.NaN()
	if _, err = JointAimCandidateInputs(f, c, h, [2]float64{}); err == nil {
		t.Fatal("nonfinite input accepted")
	}
	p := &PPO{}
	if _, _, _, err = p.jointAimSample(Observation{}, Sample{AimCandidate: &AimCandidateSample{}}); err == nil {
		t.Fatal("legacy policy accepted joint event")
	}
	if validateAimCandidates(PPOFile{}) != nil {
		t.Fatal("legacy metadata rejected")
	}
	if validateAimCandidates(PPOFile{ActionDistribution: JointAimCandidateVersion}) == nil {
		t.Fatal("missing head accepted")
	}
	if validateAimCandidates(PPOFile{AimCandidateHead: &AimCandidateFile{}}) == nil {
		t.Fatal("unversioned head accepted")
	}
	b, _ := json.Marshal(PPOFile{})
	var fields map[string]any
	_ = json.Unmarshal(b, &fields)
	if _, ok := fields["action_distribution"]; ok {
		t.Fatal("legacy canonical JSON changed")
	}
	if _, ok := fields["aim_candidate_head"]; ok {
		t.Fatal("legacy canonical JSON changed")
	}
}
