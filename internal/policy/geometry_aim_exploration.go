package policy

import (
	"fmt"
	"math"
	"reflect"
)

const ExecutionGeometryAimExplore = "geometry_aim_explore_v1"
const GeometryAimExploreVersion = "combat_geometry_aim_exploration_v1"

type GeometryAimExploreFile struct {
	Version             string    `json:"version"`
	Offsets             []float64 `json:"offset_degrees"`
	BaselineProbability float64   `json:"baseline_probability"`
	Sampler             string    `json:"sampler"`
}
type GeometryAimExploreSample struct {
	SourceLatent [2]float64 `json:"source_latent"`
	Candidate    int        `json:"candidate"`
	Candidates   int        `json:"candidates"`
}

// A separate deterministic hash stream leaves all actor RNG draws untouched.
// Avalanche mixing avoids coupling a gate to alternating weapon/frame phases.
func mixAimWord(x uint64) uint64 {
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

func GeometryAimExploreCandidate(seed int64, frame, life int) int {
	x := uint64(seed) ^ uint64(frame)*0x9e3779b97f4a7c15 ^ uint64(life)*0xd1b54a32d192ed03
	word := mixAimWord(x)
	if word>>63 == 0 {
		return 0
	}
	return 1 + int(mixAimWord(word^0x8cb92baa3f3d8dd7)%24)
}

func validateGeometryAimExplore(f PPOFile) error {
	s := f.GeometryAimExplore
	if s == nil || s.Version != GeometryAimExploreVersion || s.Sampler != "splitmix64_frame_v1" || s.BaselineProbability != .5 ||
		!reflect.DeepEqual(s.Offsets, []float64{0, -5, -2.5, 2.5, 5}) || f.Features != EquippedWeaponFeatureVersion || f.GeometryAimRanker != nil {
		return fmt.Errorf("invalid predictive aim exploration contract")
	}
	return nil
}

// Predictive TRAIN acquisition only. This is not the standard Gaussian PPO
// sampler, and no hit predictor or native outcome chooses the intervention.
func (p *PPO) exploreGeometryAim(o Observation, s *Sample) error {
	spec := p.file.GeometryAimExplore
	if spec == nil || equippedWeaponIndex(o.Weapon) != 4 || s.Target == 0 {
		return nil
	}
	f, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		return err
	}
	if f[466+(s.Target-1)*40+36] <= 0 {
		return nil
	}
	index := GeometryAimExploreCandidate(p.file.SamplingSeed, o.Identity.Frame, o.Identity.Life)
	s.AimExploration = &GeometryAimExploreSample{SourceLatent: [2]float64{s.Latent[2], s.Latent[3]}, Candidate: index, Candidates: 25}
	if index == 0 {
		return nil
	}
	scale := 180.
	if s.AimMode == 1 {
		scale = PrecisionFineDegrees
	}
	for axis, offset := range []float64{spec.Offsets[index/5], spec.Offsets[index%5]} {
		angle := math.Tanh(s.Latent[2+axis])*scale + offset
		s.Latent[2+axis] = math.Atanh(math.Max(-1+1e-6, math.Min(1-1e-6, angle/scale)))
	}
	return nil
}
