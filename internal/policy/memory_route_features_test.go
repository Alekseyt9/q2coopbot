package policy

import (
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestMemoryRouteFeaturesPreservePrefixAndSlotIdentity(t *testing.T) {
	o := Observation{Identity: Identity{Frame: 20}, RememberedThreats: []RememberedThreat{{ID: 7, Class: "monster_gunner", AgeFrames: 3}, {ID: 9, Class: "monster_parasite", AgeFrames: 4}}}
	next := quake.Vec3{512, 256, 64}
	o.MemoryRoutes = &MemoryRouteObservation{Version: MemoryRouteObservationVersion, Frame: 20, Hints: []MemoryRouteHint{{ID: 9, Age: 4, Known: true, Found: true, Distance: 4096, Next: &next, Kind: 2}, {ID: 7, Age: 3}}}
	x, err := FeaturesForVersion(o, MemoryRouteFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	base, err := FeaturesForVersion(o, ProjectileTypeNeutralFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	if len(x) != MemoryRouteFeatureWidth || !reflect.DeepEqual(x[:ProjectileTypeFeatureWidth], base) {
		t.Fatal("legacy prefix changed")
	}
	tail := x[ProjectileTypeFeatureWidth:]
	if tail[0] != 1 || tail[1] != 0 || tail[9] != 1 || tail[10] != 1 || tail[12] != 1 || tail[13] != 1 || tail[14] != .5 || tail[15] != .125 || tail[16] != .125 {
		t.Fatal(tail)
	}
	for _, change := range []func(*MemoryRouteObservation){func(r *MemoryRouteObservation) { r.Frame++ }, func(r *MemoryRouteObservation) { r.Hints[0].ID = 17 }, func(r *MemoryRouteObservation) { r.Hints[0].Age++ }, func(r *MemoryRouteObservation) { r.Hints[0].Known = false }} {
		copy := *o.MemoryRoutes
		copy.Hints = append([]MemoryRouteHint(nil), copy.Hints...)
		change(&copy)
		bad := o
		bad.MemoryRoutes = &copy
		if _, err := FeaturesForVersion(bad, MemoryRouteFeatureVersion); err == nil {
			t.Fatal("invalid route metadata accepted")
		}
	}
}
