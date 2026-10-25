package trainingepisodes

import (
	"q2coopbot/internal/quake"
	"reflect"
	"strings"
	"testing"
)

func TestHabitatRosterKeepsPrimaryAndSamplesAcrossHabitats(t *testing.T) {
	d := MonsterDistribution{Version: 2, PrimaryClass: "monster_tank", MinCount: 2, MaxCount: 4, ClassPool: []string{"monster_tank", "monster_flyer", "monster_flipper"}}
	if err := d.validateHabitat(); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	counts := map[int]bool{}
	repeat := false
	for seed := 0; seed < 256; seed++ {
		a, b := sampleMonsterRoster(d, seed), sampleMonsterRoster(d, seed)
		if !reflect.DeepEqual(a, b) || a[0] != d.PrimaryClass {
			t.Fatal("primary or seed changed")
		}
		counts[len(a)] = true
		classes := map[string]bool{}
		for _, class := range a {
			seen[class] = true
			if classes[class] {
				repeat = true
			}
			classes[class] = true
		}
	}
	if len(seen) != 3 || len(counts) != 3 || !repeat {
		t.Fatal("missing habitat, count or repetition coverage", seen, counts)
	}
	if (MonsterDistribution{Version: 2, PrimaryClass: "monster_jorg", MinCount: 1, MaxCount: 1}).validateHabitat() != nil {
		t.Fatal("explicit singleton rejected")
	}
	d.MinCount = 1
	if d.validateHabitat() == nil {
		t.Fatal("optional solo/group cannot be dispatched with fixed Mixed flag")
	}
}

func TestHabitatRosterBindsNativePoolAndPerClassPlacements(t *testing.T) {
	d := MonsterDistribution{Version: 2, PrimaryClass: "monster_tank", MinCount: 2, MaxCount: 4, ClassPool: []string{"monster_flyer", "monster_flipper"}}
	inv := DefaultPartialInventoryDistribution()
	ep := Episode{Map: "base1", Revision: 1, Monsters: habitatRosterClasses(d), Recipe: Recipe{Runner: "combat-baseline", Loadout: "weapons", Mixed: true}}
	g := Generator{Version: 1, Kind: nativeHabitatRosterKind, Roster: &d, Inventory: &inv, Site: &CampaignSite{Map: "base1", BSPSHA256: strings.Repeat("a", 64), WallDistances: make([]float64, 8)}, Distributions: map[string]StartDistribution{}}
	for i, split := range splitNames {
		p := quake.Vec3{float64(i * 10), 0, 24}
		g.Distributions[split] = StartDistribution{Player: PositionRange{Min: p, Max: p}, Primary: PositionRange{Min: p, Max: p}, Health: []int{100}, AdditionalByClass: map[string][]PositionRange{"monster_flyer": {{Min: p, Max: p}}, "monster_flipper": {{Min: p, Max: p}}}}
	}
	ep.Generator = &g
	if err := g.validate(ep); err != nil {
		t.Fatal(err)
	}
	world := quake.MapInfo{Entities: []quake.MapEntity{{Class: "monster_tank"}, {Class: "monster_flyer"}}}
	if _, err := generate(ep, "train", 1, &world); err == nil {
		t.Fatal("non-native water class accepted")
	}
	delete(g.Distributions["train"].AdditionalByClass, "monster_flipper")
	if g.validate(ep) == nil {
		t.Fatal("missing water placement accepted")
	}
}
