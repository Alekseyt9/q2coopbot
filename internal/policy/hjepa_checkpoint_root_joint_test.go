package policy

import (
	"reflect"
	"testing"

	"q2coopbot/internal/quake"
)

func TestHJEPARootJointAllFactorsAndCommonFuture(t *testing.T) {
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
	native.Frame, native.Spawncount = 1, native.Spawncount+7
	branches := make([]*HJEPA, 33)
	for i := range branches {
		branches[i], _ = LoadHJEPA(path)
		fork, err := state.WithStream(int64(i), HJEPASamplingRootJointScope)
		if err != nil || fork.Version != HJEPASamplingRootJointCheckpointVersion {
			t.Fatal(fork, err)
		}
		if err := branches[i].RestoreSamplingCheckpoint(fork, native); err != nil {
			t.Fatal(err)
		}
	}
	poses, fires, weapons := map[int]bool{}, map[bool]bool{}, map[int]bool{}
	for step := 0; step < 64; step++ {
		original := o
		original.Identity.Frame += step
		want, err := source.Decide(original)
		if err != nil {
			t.Fatal(err)
		}
		obs := original
		obs.Identity = native
		obs.Identity.Frame += step
		want.Identity = obs.Identity
		for i, p := range branches {
			got, err := p.Decide(obs)
			if err != nil {
				t.Fatal(err)
			}
			if step > 0 || i == 0 {
				if got != want || !reflect.DeepEqual(p.LastSample(), source.LastSample()) {
					t.Fatal("root joint changed common future or stream0", step, i)
				}
			} else {
				if p.LastSample().Latent == source.LastSample().Latent {
					t.Fatal("joint root did not vary movement/aim draws")
				}
				poses[p.LastSample().Vertical] = true
				fires[got.Attack] = true
				weapons[p.LastSample().Weapon] = true
			}
		}
	}
	if len(poses) != 3 || len(fires) != 2 || len(weapons) < 2 {
		t.Fatal("joint factors not exercised", poses, fires, weapons)
	}
	next := native
	next.Frame = 65
	nested, err := branches[1].CaptureSamplingCheckpoint(next)
	if err != nil || nested.Version != HJEPASamplingCheckpointVersion || nested.Scope != "" || nested.Stream != 0 || nested.Source.Frame != 244 {
		t.Fatal("root joint re-applied at nested save", nested, err)
	}
	bad := state
	bad.Scope = HJEPASamplingRootJointScope
	if bad.Validate() == nil {
		t.Fatal("new scope accepted under legacy version")
	}
}
