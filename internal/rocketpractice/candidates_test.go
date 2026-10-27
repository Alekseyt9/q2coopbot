package rocketpractice

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func fixture(t *testing.T) (policy.Observation, *quake.MapInfo) {
	t.Helper()
	root, _ := filepath.Abs(filepath.Join("..", ".."))
	world, err := quake.LoadMap(filepath.Join(root, "workspace/runtime/q2go/baseq2"), "base1")
	if err != nil {
		t.Skipf("local BSP required: %v", err)
	}
	clear := true
	solid := uint16(8290)
	o := policy.Observation{Version: policy.ObservationVersion, Identity: policy.Identity{Map: "base1", Frame: 100}, Health: 100, Position: quake.Vec3{-1632, 1288, 120.125}, Enemies: []policy.Enemy{
		{ID: 2, Relative: quake.Vec3{-192, 0, 0}, Distance: 192, ClearShot: &clear, Solid: &solid},
		{ID: 3, Relative: quake.Vec3{-118, 0, 0}, Distance: 118, ClearShot: &clear, Solid: &solid},
	}}
	return o, &world
}

func TestObservedRocketCandidateGeometryAndOrder(t *testing.T) {
	o, world := fixture(t)
	a, err := Generate(o, world)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) == 0 || len(a) > MaxCandidates {
		t.Fatal("invalid candidate budget", len(a))
	}
	floors, bodies, groups := 0, 0, 0
	for _, c := range a {
		if c.Kind == "body_reference" {
			bodies++
			continue
		}
		if c.Kind == "floor" {
			floors++
		}
		if len(c.References) > 1 {
			groups++
		}
		// A surface option must be in air immediately next to real BSP,
		// rather than an invented point below/inside a brush.
		from, to := c.Point, c.Point
		for axis := range to {
			to[axis] -= c.Normal[axis] * 4
		}
		tr := world.TraceProjectile(from, to)
		if !tr.Valid || tr.StartSolid || tr.Fraction >= 1 {
			t.Fatal("not a real adjacent surface", c)
		}
		if len(c.ObservedNearby) != 2 {
			t.Fatal("lost group geometry")
		}
	}
	if bodies != 2 || floors < 1 || groups < 1 {
		t.Fatalf("missing body/floor/group options: %d/%d/%d", bodies, floors, groups)
	}
	o.Enemies[0], o.Enemies[1] = o.Enemies[1], o.Enemies[0]
	b, err := Generate(o, world)
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatal("observation order changed options", err)
	}
}

func TestObservedRocketCandidatesRejectStaleAndHiddenInputs(t *testing.T) {
	o, world := fixture(t)
	baseline, _ := Generate(o, world)
	blocked := false
	o.Enemies = append(o.Enemies, policy.Enemy{ID: 99, Relative: quake.Vec3{8, 8, 0}, Distance: 12, ClearShot: &blocked})
	got, err := Generate(o, world)
	if err != nil || !reflect.DeepEqual(baseline, got) {
		t.Fatal("nonvisible enemy affected options", err)
	}
	o.AgeMS = 301
	if _, err := Generate(o, world); err == nil {
		t.Fatal("stale observation admitted")
	}
	o.AgeMS = 0
	o.Position[0] = math.NaN()
	if _, err := Generate(o, world); err == nil {
		t.Fatal("nonfinite input admitted")
	}
	o, _ = fixture(t)
	o.Enemies[1].ID = o.Enemies[0].ID
	if _, err := Generate(o, world); err == nil {
		t.Fatal("duplicate entity admitted")
	}
	o, _ = fixture(t)
	o.Health = 0
	got, err = Generate(o, world)
	if err != nil || len(got) != 0 {
		t.Fatal("dead actor given shot candidates")
	}
}
