package policy

import "testing"

// Metadata only: numerical inference is verified separately on CUDA/native.
func TestMemoryRouteYawVersionMetadata(t *testing.T) {
	makeFile := func(version string, outputs int) PPOFile {
		f := PPOFile{Features: MemoryRouteFeatureVersion, SpatialAim: &SpatialAimFile{}}
		s := &SpatialAimFile{Version: version}
		width := MemoryRouteMovementInputWidth
		for _, out := range []int{32, 16, outputs} {
			l := DenseLayer{Bias: make([]float64, out), Weight: make([][]float64, out)}
			for j := range l.Weight {
				l.Weight[j] = make([]float64, width)
			}
			s.Layers = append(s.Layers, l)
			width = out
		}
		f.MemoryRouteMovement = s
		return f
	}
	for _, c := range []struct {
		version string
		outputs int
		valid   bool
	}{
		{MemoryRouteMovementVersion, 2, true}, {MemoryRouteMovementYawVersion, 3, true},
		{MemoryRouteMovementVersion, 3, false}, {MemoryRouteMovementYawVersion, 2, false},
		{MemoryRouteMovementYawVersion, 4, false}, {"unversioned", 3, false},
	} {
		if err := validateMemoryRouteMovement(makeFile(c.version, c.outputs)); (err == nil) != c.valid {
			t.Fatalf("version=%s outputs=%d err=%v", c.version, c.outputs, err)
		}
	}
	f := makeFile(MemoryRouteMovementYawVersion, 3)
	f.Features = ProjectileTypeFeatureVersion
	if validateMemoryRouteMovement(f) == nil {
		t.Fatal("yaw route accepted wrong feature contract")
	}
}
