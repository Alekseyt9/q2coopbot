package policy

import (
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestHJEPAWeaponContextResolvesLegacyAlias(t *testing.T) {
	a := testObservation()
	a.Weapon = "models/weapons/v_launch/tris.md2"
	b := a
	b.Weapon = "models/weapons/v_rail/tris.md2"
	x, err := FeaturesForVersion(a, WeaponFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	y, err := FeaturesForVersion(b, WeaponFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(x, y) {
		t.Fatal("fixture no longer demonstrates legacy current-weapon alias")
	}
	wx, err := HJEPAWeaponFeatures(a)
	if err != nil {
		t.Fatal(err)
	}
	wy, err := HJEPAWeaponFeatures(b)
	if err != nil {
		t.Fatal(err)
	}
	if len(wx) != 871 || !reflect.DeepEqual(wx[:845], x) || reflect.DeepEqual(wx, wy) {
		t.Fatal("weapon context changed prefix or failed to distinguish equipment")
	}
	if wx[845+6] != 1 || wy[845+9] != 1 || wx[845+12] != 1 {
		t.Fatal("wrong observed weapon/request encoding")
	}
	req := "Rocket Launcher"
	age := 4
	a.PreviousWeaponRequest = &req
	a.PreviousWeaponRequestAgeFrames = &age
	w, err := HJEPAWeaponFeatures(a)
	if err != nil || w[845+6] != 1 || w[845+12+7] != 1 || w[845+24] != 1 {
		t.Fatal("actual weapon and local request conflated", err)
	}
	a.PreviousWeaponRequestAgeFrames = nil
	if _, err := HJEPAWeaponFeatures(a); err == nil {
		t.Fatal("incomplete own request admitted")
	}
}

func TestHJEPAZeroWeaponContextPreservesActions(t *testing.T) {
	f := hTestModel()
	f.Planner = "bc"
	f.WeaponMode = "learned_masked_v1"
	f.WeaponHead = WeaponHeadVersion
	l := hTestLayer(845, 12)
	l.Bias[4] = 5
	f.WeaponActor = []DenseLayer{l}
	old, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	f.WeaponFeatures = HJEPAWeaponFeatureVersion
	expanded := hTestLayer(871, 12)
	expanded.Bias[4] = 5
	f.WeaponActor = []DenseLayer{expanded}
	current, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	o := testObservation()
	age := 0
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 5}}
	o.Inventory = &items
	o.InventoryAgeFrames = &age
	a, err := old.Decide(o)
	if err != nil {
		t.Fatal(err)
	}
	b, err := current.Decide(o)
	if err != nil || a != b {
		t.Fatal("zero-column context extension changed action", err)
	}
	f.WeaponFeatures = "unknown"
	if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("unknown weapon context admitted")
	}
	f.WeaponFeatures = HJEPAWeaponFeatureVersion
	f.WeaponActor = []DenseLayer{l}
	if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("wrong-width context head admitted")
	}
}
