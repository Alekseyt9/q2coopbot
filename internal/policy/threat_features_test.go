package policy

import (
	"math"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestThreatFeaturesKeepLegacyAndVisibleTargetMasks(t *testing.T) {
	o := Observation{RememberedThreats: []RememberedThreat{{ID: 7, Class: "monster_soldier", Relative: quake.Vec3{512, 0, 0}, AgeFrames: 100, Velocity: &quake.Vec3{400, 0, 0}}}}
	base, err := FeaturesForVersion(o, NavigationFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	next, err := FeaturesForVersion(o, ThreatFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	if len(next) != ThreatFeatureWidth || !reflect.DeepEqual(base, next[:NavigationFeatureWidth]) {
		t.Fatal("legacy prefix changed")
	}
	tail := next[NavigationFeatureWidth:]
	if tail[0] != 1 || tail[1] != .5 || tail[2+16] != 1 || tail[23] != 1 || tail[26] != 1 || tail[27] != 1 {
		t.Fatal("memory masks/type/scales", tail[:30])
	}
	if len(TargetEnemies(o)) != 0 {
		t.Fatal("remembered enemy became a visible selectable target")
	}
	for _, x := range tail[30:] {
		if x != 0 {
			t.Fatal("absent memory slot populated")
		}
	}
	for _, change := range []func(*RememberedThreat){func(e *RememberedThreat) { e.AgeFrames = -1 }, func(e *RememberedThreat) { e.AgeFrames = 201 }, func(e *RememberedThreat) { e.Relative[0] = math.NaN() }, func(e *RememberedThreat) { e.ID = 0 }} {
		bad := o
		bad.RememberedThreats = append([]RememberedThreat(nil), o.RememberedThreats...)
		change(&bad.RememberedThreats[0])
		if _, err := FeaturesForVersion(bad, ThreatFeatureVersion); err == nil {
			t.Fatal("invalid memory accepted")
		}
	}
}
