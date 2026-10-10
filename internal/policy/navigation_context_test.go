package policy

import (
	"math"
	"q2coopbot/internal/quake"
	"testing"
)

func TestNavigationFeatureSerialization(t *testing.T) {
	o := Observation{Version: ObservationVersion}
	old, err := FeaturesForVersion(o, TargetFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	blank, err := FeaturesForVersion(o, NavigationFeatureVersion)
	if err != nil || len(blank) != NavigationFeatureWidth {
		t.Fatal(len(blank), err)
	}
	for i, x := range old {
		if blank[i] != x {
			t.Fatal("changed prefix", i)
		}
	}
	for _, x := range blank[854:] {
		if x != 0 {
			t.Fatal("missing context not masked")
		}
	}
	p := quake.Vec3{0, 512, 4096}
	age := 201
	o.ViewAngles[1] = 16384
	o.Navigation = &NavigationContext{Goal: "search_last_seen", Status: "ready", GoalRelative: &p, LastTeammateAgeFrames: &age}
	got, err := FeaturesForVersion(o, NavigationFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	n := got[854:]
	if n[0] != 1 || n[3] != 1 || n[9] != 1 || n[13] != 1 || math.Abs(n[14]-1) > 1e-12 || math.Abs(n[15]) > 1e-12 || n[16] != 4 || n[17] != 0 || n[25] != 1 || n[26] != 1 {
		t.Fatal(n)
	}
	p[0] = math.Inf(1)
	if _, err = FeaturesForVersion(o, NavigationFeatureVersion); err == nil {
		t.Fatal("accepted infinite position")
	}
	p[0] = 0
	age = -1
	if _, err = FeaturesForVersion(o, NavigationFeatureVersion); err == nil {
		t.Fatal("accepted negative age")
	}
}
