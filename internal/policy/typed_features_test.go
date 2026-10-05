package policy

import (
	"math"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestMonsterModelTypes(t *testing.T) {
	for _, item := range []struct{ class, model, want string }{
		{"monster_boss3", "models/monsters/boss3/jorg/tris.md2", "jorg"},
		{"monster_boss3", "models/monsters/boss3/rider/tris.md2", "makron"},
		{"monster_boss3", "", "unknown"}, {"monster_soldier_ss", "", "soldier"},
		{"monster_parasite", "models/custom/tris.md2", "unknown"},
	} {
		if got := MonsterType(item.class, item.model); got != item.want {
			t.Fatalf("%+v: %s", item, got)
		}
	}
}

func TestTypedFeaturesPreservePrefixAndMasks(t *testing.T) {
	clear := true
	zero := quake.Vec3{}
	skin := uint32(0)
	frame := 0
	o := Observation{Enemies: []Enemy{{ID: 67, Class: "monster_parasite", Relative: quake.Vec3{100, 0, 0}, Distance: 100, ClearShot: &clear}}}
	old, _ := FeaturesForVersion(o, BBoxFeatureVersion)
	v, err := FeaturesForVersion(o, TypedFeatureVersion)
	if err != nil || len(v) != 810 || !reflect.DeepEqual(old, v[:466]) {
		t.Fatalf("prefix/width: %d %v", len(v), err)
	}
	if v[466+15] != 1 || v[466+21] != 0 || v[786] != 0 {
		t.Fatal("type/unknown masks")
	}
	o.Enemies[0].Velocity = &zero
	o.Enemies[0].Angles = &zero
	o.Enemies[0].Skin = &skin
	o.Enemies[0].Animation = &frame
	v, _ = FeaturesForVersion(o, TypedFeatureVersion)
	if v[487] != 1 || v[488] != 0 || v[489] != 0 || v[493] != 1 || v[495] != 1 || v[498] != 1 || v[499] != 0 || v[500] != 1 || v[501] != 0 {
		t.Fatal("observed zero masks", v[487:502])
	}
	velocity := quake.Vec3{0, 400, 0}
	o.Enemies[0].Velocity = &velocity
	o.ViewAngles[1] = 16384
	v, _ = FeaturesForVersion(o, TypedFeatureVersion)
	if v[488] != 1 || v[489] != 1 || math.Abs(v[490]-1) > 1e-9 || math.Abs(v[491]) > 1e-9 {
		t.Fatal("local motion", v[487:493])
	}
	o.Enemies[0].ID = 999
	again, _ := FeaturesForVersion(o, TypedFeatureVersion)
	if !reflect.DeepEqual(v, again) {
		t.Fatal("ID leaked")
	}
	velocity[0] = math.NaN()
	if _, err := FeaturesForVersion(o, TypedFeatureVersion); err == nil {
		t.Fatal("nonfinite accepted")
	}
}

func TestObservedCompositionCountsBeyondSlotsAndExcludesOcclusion(t *testing.T) {
	clear, blocked := true, false
	s := quake.Snapshot{}
	for i := 0; i < 10; i++ {
		s.Enemies = append(s.Enemies, quake.Object{ID: i, Class: "monster_parasite", ModelPath: "models/monsters/parasite/tris.md2", Origin: quake.Vec3{float64(100 - i), 0, 0}, ClearShot: &clear})
	}
	s.Enemies = append(s.Enemies, quake.Object{Class: "monster_gunner", ClearShot: &blocked})
	o := Observe(s, Identity{}, quake.UserCmd{})
	if len(o.Enemies) != 8 || o.Enemies[0].Distance != 91 || !reflect.DeepEqual(*o.Composition, []MonsterCount{{"parasite", 10}}) {
		t.Fatal("visible composition/nearest slots", o)
	}
	v, err := FeaturesForVersion(o, TypedFeatureVersion)
	if err != nil || v[786] != 1 || v[787+15] != 1.25 || v[808] != 1.25 || v[809] != 1 {
		t.Fatal("full group features", err, v[786:])
	}
	copy := cloneEnemies(o.Enemies)
	*copy[0].Skin = 8
	(*copy[0].Angles)[0] = 90
	*copy[0].Animation = 5
	if *o.Enemies[0].Skin != 0 || *o.Enemies[0].Animation != 0 || (*o.Enemies[0].Angles)[0] != 0 {
		t.Fatal("history alias")
	}
	bad := []MonsterCount{{"parasite", 1}, {"parasite", 2}}
	o.Composition = &bad
	if _, err := FeaturesForVersion(o, TypedFeatureVersion); err == nil {
		t.Fatal("duplicate composition accepted")
	}
}
