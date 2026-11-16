package policy

import (
	"math"
	"q2coopbot/internal/quake"
	"testing"
)

func TestHJEPAWeaponBehaviorLikelihood(t *testing.T) {
	f := hTestModel()
	f.Planner = "bc"
	f.WeaponMode = "learned_masked_v1"
	f.WeaponHead = WeaponHeadVersion
	f.WeaponFeatures = HJEPAWeaponFeatureVersion
	layer := hTestLayer(HJEPAWeaponFeatureWidth, 12)
	layer.Bias[1] = math.Log(3)
	layer.Bias[9] = 1000
	f.WeaponActor = []DenseLayer{layer}
	seed := int64(42)
	f.WeaponSamplingSeed = &seed
	p, e := LoadHJEPA(hWrite(t, f))
	if e != nil {
		t.Fatal(e)
	}
	o := testObservation()
	age := 0
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}}
	o.Inventory = &items
	o.InventoryAgeFrames = &age
	probs, e := p.WeaponDistribution(o)
	if e != nil {
		t.Fatal(e)
	}
	if math.Abs(probs[0]-.25) > 1e-12 || math.Abs(probs[1]-.75) > 1e-12 || probs[9] != 0 {
		t.Fatalf("mask/normalization: %v", probs)
	}
	for frame := 1; frame <= 64; frame++ {
		o.Identity.Frame = frame
		a, e := p.Decide(o)
		if e != nil {
			t.Fatal(e)
		}
		if a.Weapon != "" && a.Weapon != "Blaster" {
			t.Fatal(a)
		}
		if _, e = p.WeaponDistribution(o); e != nil {
			t.Fatal(e)
		}
		b, e := p.Decide(o)
		if e != nil || a != b {
			t.Fatal("likelihood changed sampling")
		}
	}
	age = 21
	probs, e = p.WeaponDistribution(o)
	if e != nil || probs[0] != 1 {
		t.Fatalf("stale mask: %v %v", probs, e)
	}
	for _, held := range []bool{false, true} {
		bad := f
		if held {
			bad.WeaponOptionFrames = 16
		} else {
			bad.WeaponSamplingSeed = nil
		}
		q, e := LoadHJEPA(hWrite(t, bad))
		if e != nil {
			t.Fatal(e)
		}
		if _, e = q.WeaponDistribution(o); e == nil {
			t.Fatal("non-frame-wise likelihood admitted")
		}
	}
}
