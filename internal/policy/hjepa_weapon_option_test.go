package policy

import (
	"q2coopbot/internal/quake"
	"testing"
)

func optionTestModel(t *testing.T, frames int) (*HJEPA, Observation, *[]quake.InventoryItem) {
	t.Helper()
	f := hTestModel()
	f.Planner = "bc"
	f.WeaponMode = "learned_masked_v1"
	f.WeaponHead = WeaponHeadVersion
	f.WeaponOptionFrames = frames
	l := hTestLayer(845, 12)
	l.Bias[1] = 10
	l.Bias[4] = 20
	l.Bias[2] = 30
	f.WeaponActor = []DenseLayer{l}
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	age := 0
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 2}, {Name: "Shotgun", Count: 1}, {Name: "Shells", Count: 0}}
	o := testObservation()
	o.Identity.Frame = 100
	o.Inventory = &items
	o.InventoryAgeFrames = &age
	return p, o, &items
}

func TestHJEPAWeaponOptionExpiryAndLegacy(t *testing.T) {
	for _, frames := range []int{0, 16} {
		p, o, items := optionTestModel(t, frames)
		a, err := p.Decide(o)
		if err != nil || a.Weapon != "Machinegun" {
			t.Fatal(a, err)
		}
		(*items)[4].Count = 1
		for frame := 101; frame <= 116; frame++ {
			o.Identity.Frame = frame
			a, err = p.Decide(o)
			if err != nil {
				t.Fatal(err)
			}
			expected := "Machinegun"
			if frames == 0 || frame == 116 {
				expected = "Shotgun"
			}
			if a.Weapon != expected {
				t.Fatalf("duration%d frame%d: %s", frames, frame, a.Weapon)
			}
			b, err := p.Decide(o)
			if err != nil || b != a {
				t.Fatal("same-frame replay changed option")
			}
		}
	}
}

func TestHJEPAWeaponOptionBoundaries(t *testing.T) {
	for _, boundary := range []string{"ammo", "life", "map", "gap", "rewind", "cold_capture"} {
		t.Run(boundary, func(t *testing.T) {
			p, o, items := optionTestModel(t, 16)
			if _, err := p.Decide(o); err != nil {
				t.Fatal(err)
			}
			(*items)[4].Count = 1
			o.Identity.Frame = 101
			switch boundary {
			case "ammo":
				(*items)[2].Count = 0
			case "life":
				o.Identity.Life++
			case "map":
				o.Identity.Map = "base2"
			case "gap":
				o.Identity.Frame = 102
			case "rewind":
				o.Identity.Frame = 99
			case "cold_capture":
				o.Identity.Frame = 100
				p.ResetEpisode()
			}
			a, err := p.Decide(o)
			if err != nil || a.Weapon != "Shotgun" {
				t.Fatalf("%s did not invalidate option: %v %v", boundary, a, err)
			}
		})
	}
}

func TestHJEPAWeaponOptionMetadata(t *testing.T) {
	f := hTestModel()
	f.WeaponOptionFrames = 16
	if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("option without a head admitted")
	}
	f.WeaponMode = "learned_masked_v1"
	f.WeaponHead = WeaponHeadVersion
	f.WeaponActor = []DenseLayer{hTestLayer(845, 12)}
	for _, n := range []int{-1, 65} {
		f.WeaponOptionFrames = n
		if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
			t.Fatal("invalid duration admitted")
		}
	}
}
