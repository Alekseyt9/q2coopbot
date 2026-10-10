package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func TestCombatNavigationContextKeepsObservedAndRememberedSeparate(t *testing.T) {
	last := quake.Vec3{100, 200, 24}
	age := 201
	p := Planner{World: World{Goal: "wait_for_teammate", Navigation: "ready", Snapshot: quake.Snapshot{Self: quake.Vec3{20, 40, 24}, LastTeammate: &last, TeammateAgeFrames: &age}}, goalPoint: quake.Vec3{999, 999, 999}}
	n := p.combatNavigationContext()
	if n.GoalRelative != nil || n.WaypointRelative != nil || n.LastTeammateAgeFrames == nil || *n.LastTeammateAgeFrames != 201 || *n.LastTeammateRelative != (quake.Vec3{80, 160, 0}) {
		t.Fatal(n)
	}
	p.hasGoal = true
	p.goalPoint = last
	p.routeKnown = true
	p.route = []quake.Waypoint{{Position: quake.Vec3{60, 80, 24}}}
	n = p.combatNavigationContext()
	if *n.GoalRelative != (quake.Vec3{80, 160, 0}) || *n.WaypointRelative != (quake.Vec3{40, 40, 0}) {
		t.Fatal(n)
	}
	p.World.Snapshot.TeammateAgeFrames = nil
	if p.combatNavigationContext().LastTeammateRelative != nil {
		t.Fatal("unknown age became an observed location")
	}
}

func TestCombatNavigationContextDoesNotChangeLegacyFeatures(t *testing.T) {
	// Feature serialization only; no neural inference or numeric CPU model tests.
	o := policy.Observation{Version: policy.ObservationVersion}
	for _, version := range []string{policy.FeatureVersion, policy.AimFeatureVersion, policy.BBoxFeatureVersion, policy.TypedFeatureVersion, policy.RecoilFeatureVersion, policy.WeaponFeatureVersion, policy.TargetFeatureVersion} {
		before, err := policy.FeaturesForVersion(o, version)
		if err != nil {
			t.Fatal(err)
		}
		last := quake.Vec3{120, 20, 0}
		age := 201
		o.Navigation = &policy.NavigationContext{Goal: "search_last_seen", GoalRelative: &last, LastTeammateRelative: &last, LastTeammateAgeFrames: &age}
		after, err := policy.FeaturesForVersion(o, version)
		if err != nil {
			t.Fatal(err)
		}
		if len(before) != len(after) {
			t.Fatal("legacy input shape changed", version)
		}
		for i := range before {
			if before[i] != after[i] {
				t.Fatal("legacy input changed", version, i)
			}
		}
		o.Navigation = nil
	}
}
