package policy

import (
	"testing"

	"q2coopbot/internal/quake"
)

func TestDynamicBodyObservationMasksAndClientMembership(t *testing.T) {
	solid := uint16(2 | 3<<5 | 7<<10)
	corpse := quake.Object{ID: 4, Class: "monster_infantry", ModelPath: "models/monsters/infantry/tris.md2",
		Origin: quake.Vec3{40, 0, 24}, Solid: solid, Frame: 170, Angles: quake.Vec3{0, 90, 0}, Skin: 1}
	unknown := quake.Object{ID: 5, Class: "monster_infantry", Origin: quake.Vec3{40, 0, 24}, Frame: 171}
	s := quake.Snapshot{Frame: 101, CollisionBodies: []quake.Object{corpse,
		{ID: 1, Origin: quake.Vec3{10, 0, 24}, Solid: solid},
		{ID: 7, Origin: quake.Vec3{600, 0, 24}, Solid: solid},
		{ID: 8, Origin: quake.Vec3{50, 0, -100}, Solid: solid}}, Defeated: []quake.Object{corpse, unknown}}
	result := observedDynamicBodies(1, s, func(at quake.Vec3) bool { return at[2] >= 0 })
	if result.Version != DynamicBodyObservationVersion || result.Frame != 101 || result.Truncated || len(result.Bodies) != 2 {
		t.Fatal("wrong client membership/visibility/range", result)
	}
	a, b := result.Bodies[0], result.Bodies[1]
	if a.ID != 4 || b.ID != 5 || !a.ObservedDeathAnimation || !b.ObservedDeathAnimation || a.Animation != 170 || a.ModelPath != corpse.ModelPath || a.Skin != 1 || a.Angles != corpse.Angles {
		t.Fatal("lost observed appearance or stable ordering", a, b)
	}
	if !a.ProtocolBoundsKnown || a.ProtocolMins == nil || a.ProtocolMaxs == nil || b.ProtocolBoundsKnown || b.ProtocolMins != nil || b.ProtocolMaxs != nil || b.Solid != 0 {
		t.Fatal("invented unknown corpse bounds", a, b)
	}
	*a.ProtocolMins = quake.Vec3{}
	if s.Defeated[0] != corpse || s.CollisionBodies[0] != corpse {
		t.Fatal("observation aliases snapshot state")
	}
	legacy := corpse
	legacy.Solid, legacy.ModelPath, legacy.Angles, legacy.Skin = 0, "", quake.Vec3{}, 0
	s.Defeated[0] = legacy
	merged := observedDynamicBodies(1, s, func(at quake.Vec3) bool { return at[2] >= 0 }).Bodies[0]
	if !merged.ProtocolBoundsKnown || merged.Solid != solid || merged.ModelPath != corpse.ModelPath || merged.Angles != corpse.Angles || merged.Skin != corpse.Skin {
		t.Fatal("discarded available client collision metadata", merged)
	}
	var unavailable Observation
	EnrichEnvironment(&unavailable, s, nil)
	if unavailable.DynamicBodies != nil {
		t.Fatal("invented body visibility without map geometry")
	}
}

func TestDynamicBodyObservationCapacityAndEmptyMask(t *testing.T) {
	s := quake.Snapshot{Frame: 2}
	for id := 20; id >= 1; id-- {
		s.CollisionBodies = append(s.CollisionBodies, quake.Object{ID: id, Origin: quake.Vec3{40, 0, 0}})
	}
	result := observedDynamicBodies(0, s, func(quake.Vec3) bool { return true })
	if !result.Truncated || len(result.Bodies) != dynamicBodyCapacity {
		t.Fatal("missing bounded set", result)
	}
	for i, body := range result.Bodies {
		if body.ID != i+1 {
			t.Fatal("unstable tie ordering", result.Bodies)
		}
	}
	empty := observedDynamicBodies(0, s, func(quake.Vec3) bool { return false })
	if empty == nil || empty.Bodies == nil || len(empty.Bodies) != 0 || empty.Truncated {
		t.Fatal("unknown/known-empty masks collapsed", empty)
	}
}
