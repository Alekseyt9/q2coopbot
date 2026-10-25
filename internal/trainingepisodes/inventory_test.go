package trainingepisodes

import (
	"reflect"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func randomInventoryFixture() InventoryDistribution {
	d := InventoryDistribution{Version: 1, WeaponPool: policy.WeaponChoices()[2:], MinWeapons: 0, MaxWeapons: 5, AmmoCounts: map[string][]int{}}
	for _, name := range inventoryAmmoOrder {
		d.AmmoCounts[name] = []int{0, 1, 2, 5, 10}
	}
	return d
}

func TestRandomInventoryReproduciblePartialAndAvailable(t *testing.T) {
	d := randomInventoryFixture()
	seen := map[string]bool{}
	empty := false
	armed := false
	for seed := 0; seed < 2048; seed++ {
		a, err := SampleInitialInventory(d, seed)
		if err != nil {
			t.Fatal(err)
		}
		b, err := SampleInitialInventory(d, seed)
		if err != nil || !reflect.DeepEqual(a, b) {
			t.Fatal("seed not reproducible")
		}
		if len(a.RequestedWeapons) > 5 {
			t.Fatal("partial inventory exceeded selected slots")
		}
		items := []quake.InventoryItem{}
		unique := map[string]bool{}
		for _, item := range a.Items {
			if unique[item.Name] {
				t.Fatal("shared ammo duplicated", item.Name)
			}
			unique[item.Name] = true
			items = append(items, quake.InventoryItem{Name: item.Name, Count: item.Count})
			if item.Count == 0 {
				empty = true
			} else if item.Name != "Blaster" {
				armed = true
			}
		}
		age := 0
		mask, err := policy.WeaponAvailability(policy.Observation{Inventory: &items, InventoryAgeFrames: &age})
		if err != nil {
			t.Fatal(err)
		}
		for i, name := range policy.WeaponChoices() {
			if name == a.InitialWeapon && !mask[i] {
				t.Fatal("equipped unavailable weapon")
			}
			if i > 0 && mask[i] {
				seen[name] = true
			}
		}
	}
	// This distribution intentionally has at most10cells, so BFG cannot be
	// equipped; it must still appear as an owned, ammo-starved weapon.
	if len(seen) != 10 || seen["BFG10K"] || !empty || !armed {
		t.Fatal("weapon/resource coverage inconsistent", seen)
	}
}

func TestRandomInventorySharedAmmoAndWeaponThresholds(t *testing.T) {
	for _, tc := range []struct {
		weapon, ammo string
		low, high    int
	}{{"Super Shotgun", "Shells", 1, 2}, {"BFG10K", "Cells", 49, 50}} {
		d := randomInventoryFixture()
		d.WeaponPool = []string{tc.weapon}
		d.MinWeapons = 1
		d.MaxWeapons = 1
		for _, n := range []int{tc.low, tc.high} {
			d.AmmoCounts[tc.ammo] = []int{n}
			a, err := SampleInitialInventory(d, 1)
			if err != nil {
				t.Fatal(err)
			}
			if n == tc.low && a.InitialWeapon != "Blaster" {
				t.Fatal("insufficient ammo equipped")
			}
		}
	}
	d := randomInventoryFixture()
	d.WeaponPool = []string{"Grenade Launcher"}
	d.MinWeapons = 1
	d.MaxWeapons = 1
	d.AmmoCounts["Grenades"] = []int{5}
	a, err := SampleInitialInventory(d, 17)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, item := range a.Items {
		if item.Name == "Grenades" {
			n++
			if item.Count != 5 {
				t.Fatal("grenade stack misrepresented")
			}
		}
	}
	if n != 1 {
		t.Fatal("separate hand grenade/launcher ammo stacks")
	}
}

func TestRandomInventoryRejectsUnsafeAndInvalidItems(t *testing.T) {
	for _, change := range []func(*InventoryDistribution){
		func(d *InventoryDistribution) { d.WeaponPool = []string{"Blaster"} },
		func(d *InventoryDistribution) { d.WeaponPool = []string{"Shotgun;quit"} },
		func(d *InventoryDistribution) { d.WeaponPool = []string{"Shotgun", "Shotgun"} },
		func(d *InventoryDistribution) { d.AmmoCounts["Cells"] = []int{-1} },
		func(d *InventoryDistribution) { d.AmmoCounts["Grenades"] = []int{51} },
	} {
		d := randomInventoryFixture()
		change(&d)
		if _, err := SampleInitialInventory(d, 1); err == nil {
			t.Fatal("invalid inventory admitted")
		}
	}
}
