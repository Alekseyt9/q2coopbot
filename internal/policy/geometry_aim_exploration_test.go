package policy

import "testing"

// Hash sampler/metadata only: no neural forward pass or learned scoring.
func TestGeometryAimExplorationSamplerContract(t *testing.T) {
	var counts [25]int
	for frame := 1; frame <= 10000; frame++ {
		i := GeometryAimExploreCandidate(12345, frame, 1)
		if i < 0 || i >= 25 || i != GeometryAimExploreCandidate(12345, frame, 1) {
			t.Fatal("invalid/nonrepeatable candidate")
		}
		counts[i]++
	}
	if counts[0] < 4500 || counts[0] > 5500 {
		t.Fatal("baseline gate coverage", counts)
	}
	for _, n := range counts[1:] {
		if n < 100 || n > 350 {
			t.Fatal("offset coverage", counts)
		}
	}
	if greedyExecutionCategories(ExecutionGeometryAimExplore) {
		t.Fatal("exploration changed category selection")
	}
	f := PPOFile{ExecutionMode: ExecutionGeometryAimExplore, Features: EquippedWeaponFeatureVersion, TargetHead: TargetHeadVersion, AimModeHead: PrecisionHeadVersion,
		GeometryAimExplore: &GeometryAimExploreFile{Version: GeometryAimExploreVersion, Offsets: []float64{0, -5, -2.5, 2.5, 5}, BaselineProbability: .5, Sampler: "splitmix64_frame_v1"}}
	if err := validateExecutionMode(f); err != nil {
		t.Fatal(err)
	}
	if (&PPO{file: f}).IsStochastic() {
		t.Fatal("exploration incorrectly marked standard PPO eligible")
	}
	f.ExecutionMode = ""
	if validateExecutionMode(f) == nil {
		t.Fatal("unversioned exploration accepted")
	}
}
