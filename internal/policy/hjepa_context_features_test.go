package policy

import (
	"encoding/json"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func hContextExtension(t *testing.T, source HJEPAFile) HJEPAFile {
	t.Helper()
	b, err := json.Marshal(source)
	if err != nil {
		t.Fatal(err)
	}
	var f HJEPAFile
	if err = json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	f.Features = HJEPAContextFeatureVersion
	for i := range f.Levels[0].Encoder[0].Weight {
		f.Levels[0].Encoder[0].Weight[i] = append(f.Levels[0].Encoder[0].Weight[i], make([]float64, 26)...)
	}
	if len(f.Motor) > 0 {
		for i := range f.Motor[0].Weight {
			f.Motor[0].Weight[i] = append(f.Motor[0].Weight[i], make([]float64, 26)...)
		}
	}
	return f
}

func TestHJEPAContextZeroExtensionPreservesPlannersAndIndependentWeaponHeads(t *testing.T) {
	for _, planner := range []string{"bc", "hierarchy", "flat", "sequence"} {
		for _, head := range []string{"keep", "legacy", "context", "sampled", "held"} {
			t.Run(planner+"/"+head, func(t *testing.T) {
				f := hTestModel()
				f.Planner = planner
				f.Levels[0].Encoder[0].Weight[0][0] = .3
				motor := hTestLayer(845, 8)
				motor.Weight[0][0] = .17
				motor.Bias[4], motor.Bias[5] = -1, 1
				f.Motor = []DenseLayer{motor}
				f.WeaponMode = "keep_equipped_v1"
				if head != "keep" {
					f.WeaponMode, f.WeaponHead = "learned_masked_v1", WeaponHeadVersion
					width := 845
					if head != "legacy" {
						width = 871
						f.WeaponFeatures = HJEPAWeaponFeatureVersion
					}
					weapon := hTestLayer(width, 12)
					weapon.Weight[4][0] = .2
					if width == 871 {
						weapon.Weight[9][845+9] = .4
					}
					f.WeaponActor = []DenseLayer{weapon}
					if head == "sampled" {
						seed := int64(7)
						f.WeaponSamplingSeed = &seed
					}
					if head == "held" {
						f.WeaponOptionFrames = 3
					}
				}
				legacy, err := LoadHJEPA(hWrite(t, f))
				if err != nil {
					t.Fatal(err)
				}
				expanded, err := LoadHJEPA(hWrite(t, hContextExtension(t, f)))
				if err != nil {
					t.Fatal(err)
				}
				o := testObservation()
				items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 10}, {Name: "Railgun", Count: 1}, {Name: "Slugs", Count: 5}, {Name: "Grenade Launcher", Count: 1}, {Name: "Grenades", Count: 4}}
				age := 0
				o.Inventory, o.InventoryAgeFrames = &items, &age
				for frame := 0; frame < 5; frame++ {
					o.Identity.Frame++
					o.Weapon = "models/weapons/v_rail/tris.md2"
					if frame%2 == 0 {
						o.Weapon = "models/weapons/v_launch/tris.md2"
					}
					request := "Rocket Launcher"
					o.PreviousWeaponRequest = &request
					o.PreviousWeaponRequestAgeFrames = &age
					a, err := legacy.Decide(o)
					if err != nil {
						t.Fatal(err)
					}
					b, err := expanded.Decide(o)
					if err != nil || a != b {
						t.Fatal("zero extension changed action", a, b, err)
					}
					x, _ := HJEPAFeatures(o, WeaponFeatureVersion)
					wx, _ := HJEPAFeatures(o, HJEPAContextFeatureVersion)
					primitive := []float64{.1, 0, 0, 0, 0, 1, 0, 0}
					z, raw, prediction, err := legacy.Replay(x, primitive)
					if err != nil {
						t.Fatal(err)
					}
					wz, wraw, wprediction, err := expanded.Replay(wx, primitive)
					if err != nil || !reflect.DeepEqual(z, wz) || !reflect.DeepEqual(raw, wraw) || !reflect.DeepEqual(prediction, wprediction) {
						t.Fatal("zero extension changed world or motor", err)
					}
					if head == "sampled" {
						p, err := legacy.WeaponDistribution(o)
						if err != nil {
							t.Fatal(err)
						}
						q, err := expanded.WeaponDistribution(o)
						if err != nil || !reflect.DeepEqual(p, q) {
							t.Fatal("world width changed weapon likelihood", err)
						}
					}
				}
			})
		}
	}
}

func TestHJEPAContextReachesWorldAndMotor(t *testing.T) {
	f := hContextExtension(t, hTestModel())
	f.Planner = "bc"
	f.Levels[0].Encoder[0].Weight[0][845+6] = 1
	motor := hTestLayer(871, 8)
	motor.Weight[0][845+9] = 1
	motor.Weight[1][845+12+7] = .5
	motor.Bias[4], motor.Bias[5] = -1, 1
	f.Motor = []DenseLayer{motor}
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	a := testObservation()
	a.Weapon = "models/weapons/v_launch/tris.md2"
	b := a
	b.Weapon = "models/weapons/v_rail/tris.md2"
	x, _ := HJEPAFeatures(a, f.Features)
	y, _ := HJEPAFeatures(b, f.Features)
	za, _, _, err := p.Replay(x, []float64{0, 0, 0, 0, 0, 1, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	zb, _, _, err := p.Replay(y, []float64{0, 0, 0, 0, 0, 1, 0, 0})
	if err != nil || reflect.DeepEqual(za, zb) {
		t.Fatal("world ignored observed equipment", err)
	}
	aa, err := p.Decide(a)
	if err != nil {
		t.Fatal(err)
	}
	bb, err := p.Decide(b)
	if err != nil || aa.Forward == bb.Forward {
		t.Fatal("motor ignored observed equipment", err)
	}
	request := "Rocket Launcher"
	age := 1
	a.PreviousWeaponRequest, a.PreviousWeaponRequestAgeFrames = &request, &age
	requested, err := p.Decide(a)
	if err != nil || requested.Side == aa.Side {
		t.Fatal("motor ignored own request", err)
	}
}

func TestHJEPAContextRejectsMismatchedWidths(t *testing.T) {
	f := hTestModel()
	f.Features = HJEPAContextFeatureVersion
	if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("871 header accepted 845 encoder")
	}
	f = hContextExtension(t, hTestModel())
	f.Motor = []DenseLayer{hTestLayer(845, 8)}
	if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("871 header accepted 845 motor")
	}
	f = hContextExtension(t, hTestModel())
	f.Features = WeaponFeatureVersion
	if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("845 header accepted 871 encoder")
	}
	f.Features = "unknown"
	if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("unknown context admitted")
	}
	f = hContextExtension(t, hTestModel())
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err = p.Replay(make([]float64, 845), make([]float64, 8)); err == nil {
		t.Fatal("wrong replay width admitted")
	}
}
