package policy

import (
	"encoding/json"
	"math"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestHJEPACheckpointKeepsBothDrawStreamsAcrossNativeReconnect(t *testing.T) {
	f := hSamplingModel()
	f.WeaponMode, f.WeaponHead = "learned_masked_v1", WeaponHeadVersion
	f.WeaponActor = []DenseLayer{hTestLayer(845, 12)}
	seed := int64(31)
	f.WeaponSamplingSeed = &seed
	path := hWrite(t, f)
	source, _ := LoadHJEPA(path)
	restored, _ := LoadHJEPA(path)
	o := testObservation()
	o.Identity.Life = 1
	o.Identity.Frame = 180
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 20}}
	age := 0
	o.Inventory, o.InventoryAgeFrames = &items, &age
	checkpoint, err := source.CaptureSamplingCheckpoint(o.Identity)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	var loaded HJEPASamplingCheckpoint
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatal(err)
	}
	native := o.Identity
	native.Connection += 3
	native.Spawncount += 7
	native.Frame = 1
	if err := restored.RestoreSamplingCheckpoint(loaded, native); err != nil {
		t.Fatal(err)
	}
	weapons := map[int]bool{}
	for step := 0; step < 64; step++ {
		original := o
		original.Identity.Frame += step
		want, err := source.Decide(original)
		if err != nil {
			t.Fatal(err)
		}
		gotObservation := original
		gotObservation.Identity = native
		gotObservation.Identity.Frame += step
		got, err := restored.Decide(gotObservation)
		if err != nil {
			t.Fatal(err)
		}
		want.Identity = gotObservation.Identity
		if got != want || !reflect.DeepEqual(source.LastSample(), restored.LastSample()) {
			t.Fatalf("draws changed at restored native frame %d", gotObservation.Identity.Frame)
		}
		weapons[restored.LastSample().Weapon] = true
		lp, err := restored.PrimitiveLogProbability(gotObservation, *restored.LastSample())
		if err != nil || math.Abs(lp-restored.LastSample().LogProbability) > 1e-12 {
			t.Fatal(lp, err)
		}
	}
	if len(weapons) < 2 {
		t.Fatal("weapon RNG continuation was not exercised")
	}
	// A checkpoint of a restored branch keeps the oldest logical draw clock.
	native.Frame = 40
	nested, err := restored.CaptureSamplingCheckpoint(native)
	if err != nil || nested.Source.Frame != 219 || nested.Source.Spawncount != checkpoint.Source.Spawncount {
		t.Fatal("nested checkpoint lost logical origin", nested, err)
	}
}

func TestHJEPACheckpointRejectsWrongBindingAndClearsOnReset(t *testing.T) {
	path := hWrite(t, hSamplingModel())
	source, _ := LoadHJEPA(path)
	o := testObservation()
	o.Identity.Life = 1
	state, err := source.CaptureSamplingCheckpoint(o.Identity)
	if err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*HJEPASamplingCheckpoint, *Identity){
		func(s *HJEPASamplingCheckpoint, id *Identity) { s.Model = "other" },
		func(s *HJEPASamplingCheckpoint, id *Identity) { s.Version = "other" },
		func(s *HJEPASamplingCheckpoint, id *Identity) { s.Source.Frame = 0 },
		func(s *HJEPASamplingCheckpoint, id *Identity) { id.Actor++ },
		func(s *HJEPASamplingCheckpoint, id *Identity) { id.Map = "other" },
	} {
		p, _ := LoadHJEPA(path)
		s, id := state, o.Identity
		mutate(&s, &id)
		if p.RestoreSamplingCheckpoint(s, id) == nil || p.samplingRestore != nil {
			t.Fatal("bad checkpoint accepted")
		}
	}
	p, _ := LoadHJEPA(path)
	native := o.Identity
	native.Spawncount++
	native.Frame = 1
	if err := p.RestoreSamplingCheckpoint(state, native); err != nil {
		t.Fatal(err)
	}
	wrong := o
	wrong.Identity = native
	wrong.Identity.Life++
	if _, err := p.Decide(wrong); err == nil {
		t.Fatal("checkpoint crossed life")
	}
	p.ResetEpisode()
	plain, _ := LoadHJEPA(path)
	got, err := p.Decide(wrong)
	want, werr := plain.Decide(wrong)
	if err != nil || werr != nil || got != want {
		t.Fatal("reset did not restore ordinary sampling", err, werr)
	}
}
