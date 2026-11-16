package policy

import (
	"math"
	"reflect"
	"testing"

	"q2coopbot/internal/quake"
)

func TestHJEPACheckpointStreamsKeepWeightsAndLikelihoodButForkBothDraws(t *testing.T) {
	f := hSamplingModel()
	f.WeaponMode, f.WeaponHead = "learned_masked_v1", WeaponHeadVersion
	f.WeaponActor = []DenseLayer{hTestLayer(845, 12)}
	seed := int64(31)
	f.WeaponSamplingSeed = &seed
	path := hWrite(t, f)
	source, err := LoadHJEPA(path)
	if err != nil {
		t.Fatal(err)
	}
	o := testObservation()
	o.Identity.Life = 1
	o.Identity.Frame = 180
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 20}}
	age := 0
	o.Inventory, o.InventoryAgeFrames = &items, &age
	state, err := source.CaptureSamplingCheckpoint(o.Identity)
	if err != nil {
		t.Fatal(err)
	}
	state.Version, state.Stream = HJEPASamplingBranchCheckpointVersion, 1
	a, _ := LoadHJEPA(path)
	b, _ := LoadHJEPA(path)
	c, _ := LoadHJEPA(path)
	na, nb := o.Identity, o.Identity
	na.Frame = 1
	na.Spawncount += 20
	nb.Frame = 7
	nb.Connection += 3
	nb.Spawncount += 30
	if err := a.RestoreSamplingCheckpoint(state, na); err != nil {
		t.Fatal(err)
	}
	if err := b.RestoreSamplingCheckpoint(state, nb); err != nil {
		t.Fatal(err)
	}
	state.Stream = 2
	if err := c.RestoreSamplingCheckpoint(state, na); err != nil {
		t.Fatal(err)
	}
	primitiveDiff, weaponDiff := 0, 0
	for step := 0; step < 64; step++ {
		oa, ob := o, o
		oa.Identity, ob.Identity = na, nb
		oa.Identity.Frame += step
		ob.Identity.Frame += step
		want, err := a.Decide(oa)
		if err != nil {
			t.Fatal(err)
		}
		got, err := b.Decide(ob)
		if err != nil {
			t.Fatal(err)
		}
		want.Identity = got.Identity
		if want != got || !reflect.DeepEqual(a.LastSample(), b.LastSample()) {
			t.Fatal("same stream changed across native sessions")
		}
		other, err := c.Decide(oa)
		if err != nil {
			t.Fatal(err)
		}
		if a.LastSample().Latent != c.LastSample().Latent {
			primitiveDiff++
		}
		if other.Weapon != want.Weapon {
			weaponDiff++
		}
		lp, err := a.PrimitiveLogProbability(oa, *a.LastSample())
		if err != nil || math.Abs(lp-a.LastSample().LogProbability) > 1e-12 {
			t.Fatal("fork changed behavior density", lp, err)
		}
	}
	if primitiveDiff == 0 || weaponDiff == 0 || a.Version() != source.Version() || c.Version() != source.Version() {
		t.Fatal("draw fork missing or model changed")
	}
	next := na
	next.Frame += 64
	nested, err := a.CaptureSamplingCheckpoint(next)
	if err != nil {
		t.Fatal(err)
	}
	if nested.Stream != 1 || nested.Version != HJEPASamplingBranchCheckpointVersion || nested.Source.Frame != 244 {
		t.Fatal("nested fork origin lost", nested)
	}
	loaded, _ := LoadHJEPA(path)
	newRoot := nb
	newRoot.Frame = 1
	newRoot.Spawncount += 7
	if err := loaded.RestoreSamplingCheckpoint(nested, newRoot); err != nil {
		t.Fatal(err)
	}
	original, rebased := o, o
	original.Identity, rebased.Identity = next, newRoot
	want, err := a.Decide(original)
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Decide(rebased)
	want.Identity = got.Identity
	if err != nil || want != got || !reflect.DeepEqual(a.LastSample(), loaded.LastSample()) {
		t.Fatal("nested stream continuation changed", err)
	}
	for _, bad := range []HJEPASamplingCheckpoint{{Version: HJEPASamplingCheckpointVersion, Stream: 1, Model: state.Model, Source: state.Source}, {Version: HJEPASamplingBranchCheckpointVersion, Stream: -1, Model: state.Model, Source: state.Source}} {
		fresh, _ := LoadHJEPA(path)
		if fresh.RestoreSamplingCheckpoint(bad, na) == nil {
			t.Fatal("invalid stream admitted")
		}
	}
}
