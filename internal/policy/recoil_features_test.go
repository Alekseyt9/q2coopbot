package policy

import (
	"math"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestRecoilFeaturesPreserveV4AndMaskUnavailable(t *testing.T) {
	o := Observation{}
	old, e := FeaturesForVersion(o, TypedFeatureVersion)
	if e != nil {
		t.Fatal(e)
	}
	current, e := FeaturesForVersion(o, RecoilFeatureVersion)
	if e != nil || len(current) != 814 || !reflect.DeepEqual(old, current[:810]) {
		t.Fatalf("v4 prefix changed: %v", e)
	}
	if !reflect.DeepEqual(current[810:], []float64{0, 0, 0, 0}) {
		t.Fatal("invented unavailable recoil")
	}
	kick := quake.Vec3{-13.5, 1.5, -.25}
	o.KickAngles = &kick
	current, e = FeaturesForVersion(o, RecoilFeatureVersion)
	if e != nil || !reflect.DeepEqual(current[810:], []float64{1, -13.5 / 32, 1.5 / 32, -.25 / 32}) {
		t.Fatalf("bad recoil features %v %v", current[810:], e)
	}
	kick[0] = math.NaN()
	if _, e = FeaturesForVersion(o, RecoilFeatureVersion); e == nil {
		t.Fatal("accepted nonfinite recoil")
	}
	s := quake.Snapshot{KickAngles: quake.Vec3{1, 2, 3}, KickAnglesKnown: true}
	observed := Observe(s, Identity{}, quake.UserCmd{})
	if observed.KickAngles == nil || *observed.KickAngles != s.KickAngles {
		t.Fatal("observation lost kick")
	}
}
