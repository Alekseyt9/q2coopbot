package trainingepisodes

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestGeneratedConditionsArePairedAndPlanCannotBeModified(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := loadGenerationWorld(root); err != nil {
		t.Skipf("local BSP needed: %v", err)
	}
	r, err := Load(filepath.Join(root, "scripts", "scenarios", "combat-training", "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	output := t.TempDir()
	p, err := Build(r, root, []string{"parasite-blaster-generated"}, "validation", "rules", "", output, 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	again, err := Build(r, root, []string{"parasite-blaster-generated"}, "validation", "rules", "", output, 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p.Tasks[0].Instances, again.Tasks[0].Instances) {
		t.Fatal("same registered cohort generated different starts")
	}
	v := p.Tasks[0].Instances
	if len(v) != 4 || v[0].EngineSeed == v[1].EngineSeed || reflect.DeepEqual(v[0].Player, v[1].Player) && reflect.DeepEqual(v[0].Monsters, v[1].Monsters) {
		t.Fatal("independent seeds did not vary conditions")
	}
	path := filepath.Join(t.TempDir(), "plan.json")
	write(t, path, p)
	if err := VerifyPlan(path, root); err != nil {
		t.Fatal(err)
	}
	p.Tasks[0].Instances[0].Player[0] += 100
	write(t, path, p)
	if err := VerifyPlan(path, root); err == nil {
		t.Fatal("modified sampled position accepted")
	}
}

func TestEverySupportedGeneratorSplitProducesValidStarts(t *testing.T) {
	root, _ := filepath.Abs(filepath.Join("..", ".."))
	world, err := loadGenerationWorld(root)
	if err != nil {
		t.Skipf("local BSP needed: %v", err)
	}
	r, err := Load(filepath.Join(root, "scripts", "scenarios", "combat-training", "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, ep := range r.Episodes {
		if ep.Generator == nil {
			continue
		}
		for _, split := range splitNames {
			for i := 0; i < 4; i++ {
				v, err := generate(ep, split, ep.Splits[split].Start+i, &world)
				if err != nil {
					t.Fatal(err)
				}
				if checkStart(v, &world) != "" || v.GeometrySHA256 == "" || v.EngineSeed == v.GenerationSeed {
					t.Fatalf("invalid generated start: %+v", v)
				}
			}
		}
	}
}

func TestGeneratorRejectsConditionLeakageAndUnsupportedComposition(t *testing.T) {
	root, _ := filepath.Abs(filepath.Join("..", ".."))
	r, err := Load(filepath.Join(root, "scripts", "scenarios", "combat-training", "index.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ep Episode
	for _, v := range r.Episodes {
		if v.Generator != nil {
			ep = v
			break
		}
	}
	old := ep.Generator.Distributions["test"]
	ep.Generator.Distributions["test"] = ep.Generator.Distributions["train"]
	if err := ep.validate(); err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Fatalf("condition leakage accepted: %v", err)
	}
	ep.Generator.Distributions["test"] = old
	ep.Monsters = []string{"monster_soldier"}
	if err := ep.validate(); err == nil {
		t.Fatal("unimplemented composition accepted")
	}
}
