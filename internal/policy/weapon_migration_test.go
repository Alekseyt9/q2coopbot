package policy

import (
	"encoding/json"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func TestInventoryMigrationPreservesConfirmedTemporalPolicy(t *testing.T) {
	path := os.Getenv("Q2_PARITY_MODEL")
	if path == "" {
		t.Skip("set Q2_PARITY_MODEL for real checkpoint migration parity")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var model PPOFile
	if err = json.Unmarshal(data, &model); err != nil {
		t.Fatal(err)
	}
	if model.Features != RecoilFeatureVersion || model.Attention == nil {
		t.Fatal("requires confirmed Temporal v5")
	}
	model.Deterministic = true
	oldFile := filepath.Join(t.TempDir(), "old.json")
	encoded, _ := json.Marshal(model)
	os.WriteFile(oldFile, encoded, 0600)
	old, err := LoadPPO(oldFile)
	if err != nil {
		t.Fatal(err)
	}
	model.Features = WeaponFeatureVersion
	for _, net := range [][]DenseLayer{model.Actor, model.Value} {
		for i := range net[0].Weight {
			net[0].Weight[i] = append(net[0].Weight[i], make([]float64, 31)...)
		}
	}
	newFile := filepath.Join(t.TempDir(), "new.json")
	encoded, _ = json.Marshal(model)
	os.WriteFile(newFile, encoded, 0600)
	current, err := LoadPPO(newFile)
	if err != nil {
		t.Fatal(err)
	}
	items := []quake.InventoryItem{{Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 100}, {Name: "Railgun", Count: 1}, {Name: "Slugs", Count: 10}}
	age := 0
	kick := quake.Vec3{3, 0, 0}
	for i := 0; i < 65; i++ {
		o := Observation{Version: ObservationVersion, Identity: Identity{Map: "base1", Frame: i + 1, Actor: 1, Life: 1, Connection: 1, Spawncount: 1}, Health: 100, OnGround: true, Ammo: int16(100 - i), GunFrame: i % 6, KickAngles: &kick, Inventory: &items, InventoryAgeFrames: &age}
		if i%3 == 0 {
			o.Inventory = nil
		}
		age = i % 25
		a, err := old.Decide(o)
		if err != nil {
			t.Fatal(err)
		}
		b, err := current.Decide(o)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("migration changed action at frame%d", i)
		}
		av, err := old.Value(o)
		if err != nil {
			t.Fatal(err)
		}
		bv, err := current.Value(o)
		if err != nil || av != bv {
			t.Fatalf("migration changed value frame%d", i)
		}
	}
}
