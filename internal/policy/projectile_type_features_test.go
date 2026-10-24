package policy

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestProjectileTypePPOHeadersMetadata(t *testing.T) {
	root := os.Getenv("Q2_PROJECTILE_MODEL_ROOT")
	if root == "" {
		t.Skip("requires CUDA-migrated projectile model artifacts")
	}
	for _, entry := range []struct{ name, version string }{{"typed", ProjectileTypeFeatureVersion}, {"neutral", ProjectileTypeNeutralFeatureVersion}} {
		model, err := LoadPPO(filepath.Join(root, entry.name, "weights.json"))
		if err != nil {
			t.Fatal("projectile PPO metadata rejected", entry.name, err)
		}
		if model.FeatureVersion() != entry.version || !model.UsesThreatMemory() || model.CombatExecutionContract() != LegacyCombatExecution {
			t.Fatal("incorrect feature/memory/execution admission", entry.name)
		}
		if validateFeatureLayers(model.file.Actor, 81, BodyAblatedFeatureVersion) == nil {
			t.Fatal("new actor width admitted as old schema")
		}
	}
}

func TestProjectileTypeThreatMemoryMetadata(t *testing.T) {
	for _, version := range []string{ProjectileTypeFeatureVersion, ProjectileTypeNeutralFeatureVersion, ProjectileTypeAblatedFeatureVersion} {
		model := PPO{file: PPOFile{Features: version}}
		if !model.UsesThreatMemory() {
			t.Fatal("new schema must retain parent memory capability", version)
		}
	}
}

// Optional scalar feature fixture producer; never calls a neural forward pass.
func TestProjectileTypeCUDAFixtureMetadata(t *testing.T) {
	request, output := os.Getenv("Q2_PROJECTILE_TYPE_REQUEST"), os.Getenv("Q2_PROJECTILE_TYPE_OUT")
	if request == "" && output == "" {
		t.Skip("optional Go/CUDA metadata fixture")
	}
	data, err := os.ReadFile(request)
	if err != nil {
		t.Fatal(err)
	}
	var input struct {
		Observations []Observation `json:"observations"`
	}
	if err := json.Unmarshal(data, &input); err != nil || len(input.Observations) == 0 {
		t.Fatal("invalid fixture request", err)
	}
	tails, neutral := [][]float64{}, [][]float64{}
	for _, o := range input.Observations {
		parent, err := FeaturesForVersion(o, BodyAblatedFeatureVersion)
		if err != nil {
			t.Fatal(err)
		}
		value, err := FeaturesForVersion(o, ProjectileTypeFeatureVersion)
		if err != nil || len(value) != ProjectileTypeFeatureWidth || !reflect.DeepEqual(parent, value[:len(parent)]) {
			t.Fatal("fixture changed parent prefix", err)
		}
		tails = append(tails, value[len(parent):])
		control, err := FeaturesForVersion(o, ProjectileTypeNeutralFeatureVersion)
		if err != nil || !reflect.DeepEqual(parent, control[:len(parent)]) {
			t.Fatal("neutral control changed prefix", err)
		}
		neutral = append(neutral, control[len(parent):])
	}
	file, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	if err := json.NewEncoder(file).Encode(struct {
		Tails             [][]float64 `json:"tails"`
		NeutralTails      [][]float64 `json:"neutral_tails"`
		ParentPrefixExact bool        `json:"parent_prefix_exact"`
	}{tails, neutral, true}); err != nil {
		t.Fatal(err)
	}
}

func TestProjectileTypeFeaturesPreserveParentAndDisambiguate(t *testing.T) {
	o := testObservation()
	items := []Enemy{{ID: 31, Class: "grenade", Distance: 100, Relative: [3]float64{100, 0, 0}}}
	o.Projectiles = &items
	parent, err := FeaturesForVersion(o, BodyAblatedFeatureVersion)
	if err != nil {
		t.Fatal(err)
	}
	observed, err := FeaturesForVersion(o, ProjectileTypeFeatureVersion)
	if err != nil || len(observed) != ProjectileTypeFeatureWidth || !reflect.DeepEqual(parent, observed[:len(parent)]) {
		t.Fatal("parent prefix changed", err)
	}
	control, err := FeaturesForVersion(o, ProjectileTypeAblatedFeatureVersion)
	if err != nil || !reflect.DeepEqual(parent, control[:len(parent)]) {
		t.Fatal("control prefix changed", err)
	}
	if !reflect.DeepEqual(control[len(parent):], make([]float64, ProjectileTypeTailWidth)) || observed[len(parent)] != 1 {
		t.Fatal("type or ablation lost")
	}
	neutral, err := FeaturesForVersion(o, ProjectileTypeNeutralFeatureVersion)
	if err != nil || neutral[len(parent)+4] != 1 || neutral[len(parent)] != 0 {
		t.Fatal("neutral control lost slot presence", err)
	}
	items[0].Class = "rocket"
	rocketParent, err := FeaturesForVersion(o, BodyAblatedFeatureVersion)
	if err != nil || !reflect.DeepEqual(parent, rocketParent) {
		t.Fatal("old feature meaning changed", err)
	}
	rocket, err := FeaturesForVersion(o, ProjectileTypeFeatureVersion)
	if err != nil || rocket[len(parent)+2] != 1 || rocket[len(parent)] != 0 {
		t.Fatal("rocket and grenade type not distinguishable", err)
	}
}

func TestProjectileTypeSlotOrderUnknownAndNoMutation(t *testing.T) {
	o := testObservation()
	items := []Enemy{{ID: 1, Class: "blaster_bolt", Distance: 50}, {ID: 2, Class: "rocket", Distance: 10},
		{ID: 3, Class: "hand_grenade", Distance: 10}, {ID: 4, Class: "unrecognized", Distance: 20},
		{ID: 5, Class: "grenade", Distance: 100}}
	before := append([]Enemy(nil), items...)
	o.Projectiles = &items
	want := make([]float64, ProjectileTypeTailWidth)
	want[2], want[6], want[14], want[18] = 1, 1, 1, 1
	if got := projectileTypeFeatures(o); !reflect.DeepEqual(got, want) {
		t.Fatal("slot ordering/type encoding", got)
	}
	if !reflect.DeepEqual(items, before) {
		t.Fatal("input mutated")
	}
	o.Projectiles = nil
	if got := projectileTypeFeatures(o); !reflect.DeepEqual(got, make([]float64, ProjectileTypeTailWidth)) {
		t.Fatal("unavailable projectile tail not zero")
	}
}
