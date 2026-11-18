package policy

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"q2coopbot/internal/quake"
)

func TestHJEPAFireOnlyCheckpointKeepsCommonNonFireDraws(t *testing.T) {
	f := hSamplingModel()
	f.WeaponMode, f.WeaponHead = "learned_masked_v1", WeaponHeadVersion
	f.WeaponActor = []DenseLayer{hTestLayer(845, 12)}
	seed := int64(31)
	f.WeaponSamplingSeed = &seed
	path := hWrite(t, f)
	source, _ := LoadHJEPA(path)
	o := testObservation()
	o.Identity.Life, o.Identity.Frame = 1, 180
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 20}}
	age := 0
	o.Inventory, o.InventoryAgeFrames = &items, &age
	state, err := source.CaptureSamplingCheckpoint(o.Identity)
	if err != nil {
		t.Fatal(err)
	}
	native := o.Identity
	native.Connection, native.Spawncount, native.Frame = native.Connection+3, native.Spawncount+7, 1
	var models []*HJEPA
	for _, stream := range []int64{0, 1, 2} {
		p, _ := LoadHJEPA(path)
		branch, err := state.WithStream(stream, HJEPASamplingFireOnlyScope)
		if err != nil || p.RestoreSamplingCheckpoint(branch, native) != nil {
			t.Fatal(branch, err)
		}
		models = append(models, p)
	}
	differences := 0
	weapons := map[int]bool{}
	for step := 0; step < 128; step++ {
		original := o
		original.Identity.Frame += step
		want, err := source.Decide(original)
		if err != nil {
			t.Fatal(err)
		}
		obs := original
		obs.Identity = native
		obs.Identity.Frame += step
		for i, p := range models {
			got, err := p.Decide(obs)
			if err != nil {
				t.Fatal(err)
			}
			want.Identity = obs.Identity
			if i == 0 && (got != want || !reflect.DeepEqual(source.LastSample(), p.LastSample())) {
				t.Fatal("explicit fire stream zero changed legacy continuation")
			}
			expected := want
			expected.Attack = got.Attack
			a, b := source.LastSample(), p.LastSample()
			if got != expected || a.Latent != b.Latent || a.Vertical != b.Vertical || a.Weapon != b.Weapon || a.SamplingSeed != b.SamplingSeed {
				t.Fatal("fire stream changed another root draw", step, i)
			}
			lp, err := p.PrimitiveLogProbability(obs, *b)
			if err != nil || math.Abs(lp-b.LogProbability) > 1e-12 {
				t.Fatal("fire branch density changed", lp, err)
			}
		}
		if models[1].LastSample().Attack != models[2].LastSample().Attack {
			differences++
		}
		weapons[models[1].LastSample().Weapon] = true
	}
	if differences == 0 || len(weapons) < 2 {
		t.Fatal("fire differences and common sampled weapon choices were not exercised")
	}
	anchor := native
	anchor.Frame = 128
	nested, err := models[1].CaptureSamplingCheckpoint(anchor)
	if err != nil || nested.Scope != HJEPASamplingFireOnlyScope || nested.Version != HJEPASamplingFireCheckpointVersion || nested.Stream != 1 || nested.Source.Frame != 307 {
		t.Fatal("nested fire scope/clock lost", nested, err)
	}
	bytes, _ := json.Marshal(nested)
	var decoded HJEPASamplingCheckpoint
	if err := json.Unmarshal(bytes, &decoded); err != nil {
		t.Fatal(err)
	}
	p, _ := LoadHJEPA(path)
	fresh := native
	fresh.Spawncount++
	if err := p.RestoreSamplingCheckpoint(decoded, fresh); err != nil {
		t.Fatal(err)
	}
	obs := o
	obs.Identity = fresh
	if _, err := p.Decide(obs); err != nil || !reflect.DeepEqual(models[1].LastSample(), p.LastSample()) {
		t.Fatal("nested fire-only replay changed", err)
	}
}

func TestHJEPAFireScopeCannotBeInferredFromOldCheckpointVersions(t *testing.T) {
	p, _ := LoadHJEPA(hWrite(t, hSamplingModel()))
	o := testObservation()
	o.Identity.Life = 1
	state, err := p.CaptureSamplingCheckpoint(o.Identity)
	if err != nil {
		t.Fatal(err)
	}
	for _, change := range []func(*HJEPASamplingCheckpoint){
		func(s *HJEPASamplingCheckpoint) { s.Scope = HJEPASamplingFireOnlyScope },
		func(s *HJEPASamplingCheckpoint) { s.Version = HJEPASamplingFireCheckpointVersion },
		func(s *HJEPASamplingCheckpoint) { s.Version, s.Scope = HJEPASamplingFireCheckpointVersion, "unknown" },
		func(s *HJEPASamplingCheckpoint) { s.Version, s.Stream = HJEPASamplingBranchCheckpointVersion, -1 },
	} {
		bad := state
		change(&bad)
		if bad.Validate() == nil {
			t.Fatal("ambiguous/invalid checkpoint scope admitted", bad)
		}
	}
	if _, err := state.WithStream(1, "unknown"); err == nil {
		t.Fatal("unknown fork scope accepted")
	}
}
