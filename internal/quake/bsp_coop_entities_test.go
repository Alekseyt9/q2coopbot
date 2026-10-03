package quake

import "testing"

func TestCoopEntitiesExcludeOnlyUnconditionalRestrictions(t *testing.T) {
	entities := []MapEntity{{Model: 1, SpawnFlags: 1792}, {Model: 2, SpawnFlags: 4096}, {Model: 3, SpawnFlags: 2048}, {Model: 4, SpawnFlags: 256}, {Model: 5, SpawnFlags: 512}, {Model: 6, SpawnFlags: 1024}}
	got := coopMapEntities(entities)
	if len(got) != 5 || got[0].Model != 2 || got[4].Model != 6 {
		t.Fatal(got)
	}
	if len(entities) != 6 {
		t.Fatal("source entities mutated")
	}
}
