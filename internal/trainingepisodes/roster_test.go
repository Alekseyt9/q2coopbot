package trainingepisodes

import (
	"reflect"
	"testing"

	"q2coopbot/internal/quake"
)

func TestNativeRosterDeterministicRepeatedAndVariable(t *testing.T) {
	d := MonsterDistribution{Version: 1, MinCount: 2, MaxCount: 4, ClassPool: []string{"monster_soldier_light", "monster_soldier_ss"}}
	if err := d.validate(); err != nil {
		t.Fatal(err)
	}
	counts := map[int]bool{}
	repeated := false
	for seed := 0; seed < 256; seed++ {
		a, b := sampleMonsterRoster(d, seed), sampleMonsterRoster(d, seed)
		if !reflect.DeepEqual(a, b) {
			t.Fatal("same seed changed roster")
		}
		counts[len(a)] = true
		seen := map[string]bool{}
		for _, class := range a {
			if class != d.ClassPool[0] && class != d.ClassPool[1] {
				t.Fatal("foreign class", class)
			}
			if seen[class] {
				repeated = true
			}
			seen[class] = true
		}
	}
	if len(counts) != 3 || !repeated {
		t.Fatal("missing group count or repeated-class coverage")
	}
	for _, bad := range []MonsterDistribution{
		{Version: 1, MinCount: 1, MaxCount: 4, ClassPool: d.ClassPool},
		{Version: 1, MinCount: 2, MaxCount: 5, ClassPool: d.ClassPool},
		{Version: 1, MinCount: 2, MaxCount: 4, ClassPool: []string{"monster_tank"}},
		{Version: 1, MinCount: 2, MaxCount: 4, ClassPool: []string{"monster_infantry", "monster_infantry"}},
	} {
		if bad.validate() == nil {
			t.Fatal("invalid distribution accepted", bad)
		}
	}
	world := quake.MapInfo{Entities: []quake.MapEntity{{Class: "monster_soldier_light"}}}
	ep := Episode{Map: "base1", Monsters: []string{"monster_soldier_light", "monster_soldier_ss"}, Generator: &Generator{Kind: nativeRosterKind, Roster: &d}}
	if _, err := generate(ep, "train", 1, &world); err == nil {
		t.Fatal("roster accepted a non-native soldier variant")
	}
}
