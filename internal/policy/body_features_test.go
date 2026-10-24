package policy

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"q2coopbot/internal/quake"
)

func TestBodyFeaturePPOHeadersMetadata(t *testing.T) {
	root := os.Getenv("Q2_BODY_MODEL_ROOT")
	if root == "" {
		t.Skip("requires CUDA-migrated body model artifacts")
	}
	for _, test := range []struct{ name, version string }{{"observed", BodyFeatureVersion}, {"ablated", BodyAblatedFeatureVersion}} {
		model, err := LoadPPO(filepath.Join(root, test.name, "weights.json"))
		if err != nil {
			t.Fatal("body model metadata rejected", test.name, err)
		}
		if model.FeatureVersion() != test.version || model.CombatExecutionContract() != LegacyCombatExecution || !model.UsesThreatMemory() {
			t.Fatal("wrong body schema/contract", test.name)
		}
	}
}

func TestBodyFeatureThreatMemoryCapabilityMetadata(t *testing.T) {
	for _, version := range []string{FeatureVersion, AimFeatureVersion, BBoxFeatureVersion, TypedFeatureVersion, RecoilFeatureVersion, WeaponFeatureVersion, TargetFeatureVersion, NavigationFeatureVersion, ThreatFeatureVersion, EquippedWeaponFeatureVersion, BodyFeatureVersion, BodyAblatedFeatureVersion} {
		model := PPO{file: PPOFile{Features: version}}
		expected := version == ThreatFeatureVersion || version == EquippedWeaponFeatureVersion || version == BodyFeatureVersion || version == BodyAblatedFeatureVersion
		if model.UsesThreatMemory() != expected {
			t.Fatal("wrong memory capability", version)
		}
	}
}

func TestBodyFeatureSchemaPrefixAndMasks(t *testing.T) {
	o := testObservation()
	o.Identity.Frame = 101
	s := quake.Snapshot{Frame: 101, CollisionBodies: []quake.Object{{ID: 31, ModelPath: "models/monsters/infantry/tris.md2", Origin: quake.Vec3{40, 0, 24}, Solid: 7266}},
		Defeated: []quake.Object{{ID: 32, Class: "monster_infantry", ModelPath: "models/monsters/infantry/tris.md2", Origin: quake.Vec3{80, 0, 24}, Frame: 178}}}
	o.DynamicBodies = observedDynamicBodies(o.Identity.Actor, s, func(quake.Vec3) bool { return true })
	legacy, err := FeaturesForVersion(o, EquippedWeaponFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	actual, err := FeaturesForVersion(o, BodyFeatureVersion)
	if err != nil || len(actual) != BodyFeatureWidth || !reflect.DeepEqual(actual[:len(legacy)], legacy) {
		t.Fatal("changed v10 prefix or schema width", len(actual), err)
	}
	tail := actual[EquippedWeaponFeatureWidth:]
	if tail[0] != 1 || tail[2] != 1 || tail[7] != 1 || tail[22] != 1 || tail[27] != 0 || tail[34] != 1 {
		t.Fatal("body availability/shape/death masks lost", tail[:42])
	}
	abl, err := FeaturesForVersion(o, BodyAblatedFeatureVersion)
	if err != nil || !reflect.DeepEqual(abl[:len(legacy)], legacy) {
		t.Fatal("invalid body ablation", err)
	}
	for _, value := range abl[len(legacy):] {
		if value != 0 {
			t.Fatal("body data leaked into ablated tail")
		}
	}
	o.DynamicBodies.Bodies[0].ID += 100
	o.DynamicBodies.Bodies[1].ID += 100
	again, err := FeaturesForVersion(o, BodyFeatureVersion)
	if err != nil || !reflect.DeepEqual(actual, again) {
		t.Fatal("entity IDs entered body features", err)
	}
}

func TestBodyFeatureSchemaRejectsInvalidMetadata(t *testing.T) {
	o := testObservation()
	o.Identity.Frame = 101
	set := observedDynamicBodies(o.Identity.Actor, quake.Snapshot{Frame: 101,
		CollisionBodies: []quake.Object{{ID: 31, Origin: quake.Vec3{40, 0, 24}, Solid: 7266}}}, func(quake.Vec3) bool { return true })
	for _, mutate := range []func(*DynamicBodyObservation){
		func(s *DynamicBodyObservation) { s.Version = "unknown" },
		func(s *DynamicBodyObservation) { s.Frame++ },
		func(s *DynamicBodyObservation) { s.Bodies[0].Distance = math.NaN() },
		func(s *DynamicBodyObservation) { s.Bodies[0].ProtocolMins = nil },
		func(s *DynamicBodyObservation) { s.Bodies[0].Solid = 0 },
		func(s *DynamicBodyObservation) { v := quake.Vec3{-999, -999, -999}; s.Bodies[0].ProtocolMins = &v },
	} {
		copySet := *set
		copySet.Bodies = append([]ObservedDynamicBody{}, set.Bodies...)
		mutate(&copySet)
		o.DynamicBodies = &copySet
		if _, err := FeaturesForVersion(o, BodyFeatureVersion); err == nil {
			t.Fatal("accepted invalid body schema", copySet)
		}
	}
	o.DynamicBodies = nil
	if values, err := FeaturesForVersion(o, BodyFeatureVersion); err != nil || len(values) != BodyFeatureWidth {
		t.Fatal("unavailable body mask rejected", err)
	}
}
