package quake

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

func TestGroundSupportAuditPartialFootprint(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{-100, -100, -20}, {100, 100, 0}})
	m.Models = []BSPModel{{}}
	r := m.InspectGroundSupport(Vec3{105, 0, 24.125}, false, nil, nil)
	if r.Status != "recorded_geometry_only" || r.FrictionStatus != "uneven_or_missing_support" || r.Points[0].Known {
		t.Fatalf("center outside floor must remain unknown: %+v", r)
	}
	if !r.Points[1].Known || !r.Points[2].Known {
		t.Fatal("rear corners should still have support")
	}
	tr := r.Hulls[0].Trace
	if !tr.Valid || tr.StartSolid || tr.AllSolid || tr.Fraction >= 1 || tr.Normal[2] < .9999 {
		t.Fatalf("full hull should detect the remaining floor: %+v", tr)
	}
	if _, err := json.Marshal(r); err != nil {
		t.Fatal(err)
	}
}

func TestGroundSupportAuditObservedBodySupport(t *testing.T) {
	m := playerTraceBoxes([2]Vec3{{-100, -100, -84}, {100, 100, -64}})
	m.Models = []BSPModel{{}}
	bodies := []Object{{ID: 8, Origin: Vec3{0, 0, -16}, Solid: 2 | (3 << 5) | (6 << 10)}}
	before := append([]Object(nil), bodies...)
	r := m.InspectGroundSupport(Vec3{0, 0, 24.125}, false, nil, bodies)
	if len(r.Hulls) != 12 {
		t.Fatalf("missing hull diagnostics: %+v", r)
	}
	if r.Hulls[0].Trace.Fraction != 1 || r.Hulls[2].Trace.Fraction >= 1 || r.Hulls[2].Trace.Normal[2] < .9999 {
		t.Fatalf("body-only support must be distinguishable from static support: %+v", r.Hulls[:3])
	}
	if !reflect.DeepEqual(before, bodies) {
		t.Fatal("diagnostic mutated recorded bodies")
	}
}

func TestGroundSupportAuditUnknownAndInvalid(t *testing.T) {
	if r := (*MapInfo)(nil).InspectGroundSupport(Vec3{}, false, nil, nil); r.Status != "unknown_geometry" || len(r.Hulls) != 0 {
		t.Fatal(r)
	}
	m := playerTraceBoxes([2]Vec3{{-100, -100, -20}, {100, 100, 0}})
	m.Models = []BSPModel{{}}
	r := m.InspectGroundSupport(Vec3{math.NaN(), 0, 24}, false, nil, nil)
	if r.Status != "invalid_origin" || len(r.Hulls) != 0 {
		t.Fatal(r)
	}
}
