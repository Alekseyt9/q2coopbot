package policy

import (
	"math"
	"reflect"
	"testing"
)

// Pure geometry and metadata only; no learned predictor is evaluated here.
func TestObservedActionGeometryContract(t *testing.T) {
	x := make([]float64, 185)
	x[7], x[9], x[99], x[100], x[101] = 1, 1, 1, .5, .5
	x[147], x[148], x[149], x[150], x[152], x[161], x[181] = 1, .25, -24./64, 32./64, 1, 1, 1
	a, err := ObservedActionGeometry(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 29 || math.Abs(a[7]) > 1e-6 || math.Abs(a[8]) > 1e-6 || a[15] != 1 {
		t.Fatalf("bad eye geometry %v", a)
	}
	if math.Abs(a[20]-math.Atan2(8, 256)/math.Pi) > 1e-6 {
		t.Fatal("muzzle parallax")
	}
	x[6], x[7] = 1, 0
	b, _ := ObservedActionGeometry(x)
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-6 {
			t.Fatal("yaw-local frame rotation", i, a[i], b[i])
		}
	}
	x[164] = 5. / 180
	c, _ := ObservedActionGeometry(x)
	if c[15] != 0 || c[7] >= 0 {
		t.Fatal("off-axis ray must miss bbox")
	}
	x[147] = 0
	c, _ = ObservedActionGeometry(x)
	if !reflect.DeepEqual(c, make([]float64, 29)) {
		t.Fatal("unknown bbox invented geometry")
	}
	x[147] = 1
	x[164] = math.Atan2(16, 240) / math.Pi
	soft, err := ObservedActionGeometryForVersion(x, ActionGeometrySmoothVersion)
	if err != nil || math.Abs(soft[15]-.5) > .0002 {
		t.Fatal("unstable grazing-ray soft intersection", soft, err)
	}
	if _, err := ObservedActionGeometry(x[:184]); err == nil {
		t.Fatal("bad width accepted")
	}
	x[0] = math.NaN()
	if _, err := ObservedActionGeometry(x); err == nil {
		t.Fatal("nonfinite input accepted")
	}
	if greedyExecutionCategories(ExecutionGeometryAimRanker) {
		t.Fatal("aim ranking must preserve target/mode sampling")
	}
	if validateExecutionMode(PPOFile{GeometryAimRanker: &GeometryAimRankerFile{}}) == nil {
		t.Fatal("ranker accepted outside its execution mode")
	}
}

func TestCenteredMachinegunGeometryContract(t *testing.T) {
	x := make([]float64, 185)
	x[7], x[9], x[99], x[101], x[147], x[148], x[149], x[150], x[181] = 1, 1, 1, .5, 1, .25, -.375, .5, 1
	a, err := ObservedActionGeometryForVersion(x, ActionGeometryCenteredVersion)
	if err != nil || math.Abs(a[20]) > 1e-9 {
		t.Fatal("centered muzzle invented lateral parallax", a, err)
	}
	x[184] = .5 // Roll changes lateral projection only for historical right-hand muzzle.
	b, err := ObservedActionGeometryForVersion(x, ActionGeometryCenteredVersion)
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatal("centered muzzle depends on roll", a, b, err)
	}
	c, err := ObservedActionGeometryForVersion(x, ActionGeometrySmoothVersion)
	if err != nil || math.Abs(c[20]) < .001 {
		t.Fatal("historical right-hand geometry changed", c, err)
	}
}
