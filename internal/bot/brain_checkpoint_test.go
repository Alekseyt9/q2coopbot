package bot

import (
	"encoding/json"
	"q2coopbot/internal/quake"
	"testing"
)

func TestPlannerCheckpointBranchIsolationAndObservationAge(t *testing.T) {
	death := quake.Vec3{100, 200, 30}
	s := quake.Snapshot{Map: "base2", Frame: 1000, Health: 38, OnGround: true}
	p := &Planner{World: World{Snapshot: s, Goal: "return_to_death"}, deathPoint: &death, respawnRegroup: &respawnRegroup{target: death}, hasGoal: true, goalPoint: death, resources: map[int]*ResourceMemory{7: {Item: quake.Object{ID: 7, Class: "item_health", Origin: quake.Vec3{300, 400, 30}}, LastSeen: 990, State: "observed"}, 8: {Item: quake.Object{ID: 8, Class: "item_armor_body", Origin: death}, LastSeen: 995, State: "unavailable", Attempted: true}}}
	state, err := p.CaptureCheckpoint()
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	var loaded PlannerCheckpoint
	if err = json.Unmarshal(data, &loaded); err != nil {
		t.Fatal(err)
	}
	fresh := quake.Snapshot{Map: "base2", Frame: 2, Health: 38, OnGround: true}
	a, b := &Planner{}, &Planner{}
	for _, branch := range []*Planner{a, b} {
		if err = branch.RestoreCheckpoint(loaded, fresh, ""); err != nil {
			t.Fatal(err)
		}
	}
	if a.resources[7].LastSeen != -8 || a.resources[7].State != "unknown" || !a.resources[8].Attempted || a.resources[8].State != "unavailable" || a.respawnRegroup == nil || len(a.World.Route) > 0 {
		t.Fatal(a.resources)
	}
	a.resources[7].Item.Origin[0] = 999
	a.deathPoint[0] = 999
	if b.resources[7].Item.Origin[0] != 300 || b.deathPoint[0] != 100 || loaded.DeathPoint[0] != 100 {
		t.Fatal("parallel branch mutated source")
	}
	future := p.resources[7]
	future.LastSeen = 1001
	state, err = p.CaptureCheckpoint()
	if err != nil || len(state.Resources) != 1 {
		t.Fatal("future observation saved", state, err)
	}
	loaded.Map = "base1"
	if (&Planner{}).RestoreCheckpoint(loaded, fresh, "") == nil {
		t.Fatal("wrong map accepted")
	}
	loaded.Map = "base2"
	loaded.Resources[0].Age = -1
	if (&Planner{}).RestoreCheckpoint(loaded, fresh, "") == nil {
		t.Fatal("future age accepted")
	}
	p.World.Snapshot.Weapon = "Grenades"
	p.World.Snapshot.GunFrame = 11
	if _, err = p.CaptureCheckpoint(); err == nil {
		t.Fatal("armed fuse saved")
	}
}
