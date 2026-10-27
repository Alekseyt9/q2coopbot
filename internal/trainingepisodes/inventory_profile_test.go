package trainingepisodes

import (
	"reflect"
	"testing"
)

func TestInventoryProfilePacksPreserveOtherConditions(t *testing.T) {
	seen := map[int]bool{}
	for seed := 0; seed < 512; seed++ {
		original, err := SampleInitialInventory(DefaultPartialInventoryDistribution(), seed)
		if err != nil {
			t.Fatal(err)
		}
		p := &Plan{Tasks: []Task{{Split: "train", Instances: []Instance{{EngineSeed: seed, Inventory: &original}}}}}
		if err := ApplyInventoryProfile(p, BulletPacks50Profile); err != nil {
			t.Fatal(err)
		}
		got := p.Tasks[0].Instances[0].Inventory
		if got.InitialWeapon != original.InitialWeapon || !reflect.DeepEqual(got.RequestedWeapons, original.RequestedWeapons) {
			t.Fatal("changed starting weapon availability")
		}
		for i, item := range got.Items {
			if item.Name == "Bullets" {
				if item.Count%50 != 0 || item.Count > 200 || (item.Count == 0) != (original.Items[i].Count == 0) {
					t.Fatal("invalid bullet packs or changed empty/nonempty status")
				}
				seen[item.Count] = true
			} else if item != original.Items[i] {
				t.Fatal("changed another ammo or weapon")
			}
		}
		replayed := &Plan{Tasks: []Task{{Split: "train", Instances: []Instance{{EngineSeed: seed, Inventory: &original}}}}}
		if err := ApplyInventoryProfile(replayed, BulletPacks50Profile); err != nil || !reflect.DeepEqual(p, replayed) {
			t.Fatal("profile not reproducible", err)
		}
	}
	if len(seen) != 5 {
		t.Fatal("missing pack quantity", seen)
	}
}

func TestInventoryProfileBoundaryAndRocket(t *testing.T) {
	inv, _ := SampleInitialInventory(DefaultPartialInventoryDistribution(), 1)
	original, _ := SampleInitialInventory(DefaultPartialInventoryDistribution(), 1)
	p := &Plan{Tasks: []Task{{Split: "train", Instances: []Instance{{EngineSeed: 1, Inventory: &inv}}}}}
	if ApplyInventoryProfile(p, AmmoBoundaryProfile) == nil || p.InventoryProfile != "" {
		t.Fatal("boundary profile silently admitted TRAIN")
	}
	p.Tasks[0].Split = "validation"
	if err := ApplyInventoryProfile(p, AmmoBoundaryProfile); err != nil || !reflect.DeepEqual(*p.Tasks[0].Instances[0].Inventory, inv) {
		t.Fatal("boundary conditions changed", err)
	}
	if ApplyInventoryProfile(p, BulletPacks50Profile) == nil {
		t.Fatal("profile reapplied")
	}
	p.InventoryProfile = ""
	if err := ApplyInventoryProfile(p, RocketPracticeProfile); err != nil {
		t.Fatal(err)
	}
	got := p.Tasks[0].Instances[0].Inventory
	if got.InitialWeapon != "Rocket Launcher" || !reflect.DeepEqual(got.RequestedWeapons, []string{"Rocket Launcher"}) {
		t.Fatal("rocket practice did not start armed")
	}
	for _, item := range got.Items {
		switch item.Name {
		case "Blaster", "Rocket Launcher":
			if item.Count != 1 {
				t.Fatal(item)
			}
		case "Rockets":
			if item.Count != 5 && item.Count != 10 && item.Count != 20 {
				t.Fatal(item)
			}
		default:
			if item.Count != 0 {
				t.Fatal(item)
			}
		}
	}
	if !reflect.DeepEqual(inv, original) {
		t.Fatal("source inventory mutated")
	}
}
