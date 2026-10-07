package policy

import (
	"encoding/json"
	"os"
	"os/exec"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestWeaponInventoryUnknownFreshnessAndAmmoThresholds(t *testing.T) {
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Super Shotgun", Count: 1}, {Name: "BFG10K", Count: 1}, {Name: "Bullets", Count: 1}, {Name: "Shells", Count: 1}, {Name: "Cells", Count: 49}}
	age := 20
	o := Observation{Inventory: &items, InventoryAgeFrames: &age}
	old, _ := FeaturesForVersion(o, RecoilFeatureVersion)
	v, err := FeaturesForVersion(o, WeaponFeatureVersion)
	if err != nil || len(v) != 845 || !reflect.DeepEqual(v[:814], old) {
		t.Fatalf("prefix/width: %v", err)
	}
	mask, err := WeaponAvailability(o)
	if err != nil || !mask[0] || !mask[1] || !mask[4] || mask[3] || mask[10] {
		t.Fatalf("stock minima %v %v", mask, err)
	}
	items[5].Count = 2
	items[6].Count = 50
	mask, _ = WeaponAvailability(o)
	if !mask[3] || !mask[10] {
		t.Fatal("exact SSG/BFG ammo thresholds rejected")
	}
	for _, a := range []int{-1, 21} {
		age = a
		mask, _ = WeaponAvailability(o)
		if !reflect.DeepEqual(mask, []bool{true, false, false, false, false, false, false, false, false, false, false, false}) {
			t.Fatal("stale inventory exposed selectable weapons")
		}
	}
	o.InventoryAgeFrames = nil
	mask, _ = WeaponAvailability(o)
	if mask[4] {
		t.Fatal("invented inventory freshness")
	}
	o.Inventory = nil
	v, _ = FeaturesForVersion(o, WeaponFeatureVersion)
	for i, x := range v[814:] {
		if x != 0 && i != 19 {
			t.Fatal("invented unknown inventory")
		}
	}
	items = append(items, items[0])
	o.Inventory = &items
	age = 0
	o.InventoryAgeFrames = &age
	if _, err = WeaponAvailability(o); err == nil {
		t.Fatal("duplicate inventory accepted")
	}
	items = items[:len(items)-1]
	items[0].Count = -1
	if _, err = WeaponAvailability(o); err == nil {
		t.Fatal("negative inventory accepted")
	}
}

// Optional host parity test executes the independent Python encoder on known,
// empty, stale and unknown inventories, including exact stock ammo boundaries.
func TestWeaponInventoryPythonParity(t *testing.T) {
	python := os.Getenv("Q2_PARITY_PYTHON")
	if python == "" {
		t.Skip("set Q2_PARITY_PYTHON for host Go/Python parity")
	}
	items := []quake.InventoryItem{{Name: "Railgun", Count: 1}, {Name: "Slugs", Count: 1}, {Name: "Grenades", Count: 3}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 901}}
	empty := []quake.InventoryItem{}
	fresh := 0
	thresholds := []quake.InventoryItem{{Name: "Super Shotgun", Count: 1}, {Name: "Shells", Count: 1}, {Name: "BFG10K", Count: 1}, {Name: "Cells", Count: 49}}
	exact := append([]quake.InventoryItem(nil), thresholds...)
	exact[1].Count, exact[3].Count = 2, 50
	cases := []Observation{{}, {Inventory: &empty, InventoryAgeFrames: &fresh}, {Inventory: &items}, {Inventory: &thresholds, InventoryAgeFrames: &fresh}, {Inventory: &exact, InventoryAgeFrames: &fresh}}
	for _, age := range []int{0, 20, 21, -1} {
		age := age
		cases = append(cases, Observation{Inventory: &items, InventoryAgeFrames: &age})
	}
	for i, o := range cases {
		v, err := FeaturesForVersion(o, WeaponFeatureVersion)
		if err != nil {
			t.Fatal(err)
		}
		data, _ := json.Marshal(o)
		cmd := exec.Command(python, "-c", "import json,sys;sys.path.insert(0,'../../scripts');from combat_weapon_features import inventory_tail;print(json.dumps(inventory_tail(json.loads(sys.argv[1]))))", string(data))
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("parity oracle: %v %s", err, out)
		}
		var tail []float64
		if err = json.Unmarshal(out, &tail); err != nil || !reflect.DeepEqual(v[814:], tail) {
			t.Fatalf("Go/Python mismatch case %d: %v", i, err)
		}
	}
}
