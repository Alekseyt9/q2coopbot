package policy

import (
	"encoding/json"
	"reflect"
	"testing"

	"q2coopbot/internal/quake"
)

func checkpointObservation(step int) Observation {
	o := testObservation()
	o.Identity.Life = 1
	o.Identity.Frame = 170 + step
	o.Position = quake.Vec3{float64(step), 10, 20}
	o.Velocity = quake.Vec3{10, float64(step % 3), 0}
	o.Weapon = "models/weapons/v_handgr/tris.md2"
	o.GunFrame = 11
	o.OnGround = step%2 == 0
	if step%3 == 0 {
		o.PreviousCommand.Buttons = 1
	}
	clear := true
	if step < 10 || step >= 20 {
		id := 2
		if step >= 20 {
			id = 3
		}
		o.Enemies = []Enemy{{ID: id, Class: "monster_soldier", Relative: quake.Vec3{100 + float64(step), 4, 0}, Distance: 110, ClearShot: &clear}}
	}
	projectiles := []Enemy{{ID: 500, Class: "grenade", Relative: quake.Vec3{30 + float64(step), 0, 10}, Distance: 40}}
	o.Projectiles = &projectiles
	items := []quake.InventoryItem{{Name: "Grenades", Count: 4}, {Name: "Blaster", Count: 1}}
	age := 0
	o.Inventory, o.InventoryAgeFrames = &items, &age
	return o
}

func TestObservationCheckpointNativeClockRebasePreservesFeaturesAndDraws(t *testing.T) {
	var history History
	var memory ThreatMemory
	f := hSamplingModel()
	f.Features = HJEPAContextFeatureVersion
	f.Levels[0].Encoder = []DenseLayer{hTestLayer(HJEPAWeaponFeatureWidth, 2)}
	f.WeaponMode, f.WeaponHead, f.WeaponFeatures = "learned_masked_v1", WeaponHeadVersion, HJEPAWeaponFeatureVersion
	f.WeaponActor = []DenseLayer{hTestLayer(HJEPAWeaponFeatureWidth, 12)}
	seed := int64(71)
	f.WeaponSamplingSeed = &seed
	path := hWrite(t, f)
	source, loadError := LoadHJEPA(path)
	if loadError != nil {
		t.Fatal(loadError)
	}
	var root Observation
	for step := 0; step <= 7; step++ {
		root = checkpointObservation(step)
		history.Enrich(&root)
		memory.Enrich(&root, nil)
		if _, err := source.Decide(root); err != nil {
			t.Fatal(err)
		}
	}
	state, err := CaptureObservationMemory(&history, &memory, root.Identity)
	if err != nil {
		t.Fatal(err)
	}
	bytes, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	var loaded ObservationMemoryCheckpoint
	if err := json.Unmarshal(bytes, &loaded); err != nil {
		t.Fatal(err)
	}
	native := root.Identity
	native.Connection++
	native.Spawncount += 20
	native.Frame = 1
	restoredHistory, restoredMemory, err := RestoreObservationMemory(loaded, native)
	if err != nil {
		t.Fatal(err)
	}
	if restoredHistory.frames[0].Identity.Frame != -2 {
		t.Fatal("lost pre-load history ages")
	}
	sampling, err := source.CaptureSamplingCheckpoint(root.Identity)
	if err != nil {
		t.Fatal(err)
	}
	restored, loadError := LoadHJEPA(path)
	if loadError != nil {
		t.Fatal(loadError)
	}
	if err := restored.RestoreSamplingCheckpoint(sampling, native); err != nil {
		t.Fatal(err)
	}
	for step := 8; step < 45; step++ {
		a, b := checkpointObservation(step), checkpointObservation(step)
		b.Identity = native
		b.Identity.Frame += step - 7
		var defeated []quake.Object
		if step == 15 {
			defeated = []quake.Object{{ID: 2, Class: "monster_soldier"}}
		}
		history.Enrich(&a)
		memory.Enrich(&a, defeated)
		restoredHistory.Enrich(&b)
		restoredMemory.Enrich(&b, defeated)
		fa, err := HJEPAFeatures(a, HJEPAContextFeatureVersion)
		if err != nil {
			t.Fatal(err)
		}
		fb, err := HJEPAFeatures(b, HJEPAContextFeatureVersion)
		if err != nil || !reflect.DeepEqual(fa, fb) || !reflect.DeepEqual(a.RememberedThreats, b.RememberedThreats) {
			t.Fatalf("observation memory changed at step%d: %v", step, err)
		}
		want, err := source.Decide(a)
		if err != nil {
			t.Fatal(err)
		}
		got, err := restored.Decide(b)
		want.Identity = b.Identity
		if err != nil || got != want || !reflect.DeepEqual(source.LastSample(), restored.LastSample()) {
			t.Fatalf("continuation changed at step%d: %v", step, err)
		}
	}
	// Captures are private copies even after the source buffers/maps advance.
	after, _ := json.Marshal(state)
	if string(after) != string(bytes) {
		t.Fatal("source mutation changed checkpoint")
	}
	loaded.Frames[0].Enemies[0].Relative[0] = 9999
	if restoredHistory.frames[0].Enemies != nil && restoredHistory.frames[0].Enemies[0].Relative[0] == 9999 {
		t.Fatal("restore aliases serialized memory")
	}
}

func TestObservationCheckpointRejectsBrokenTemporalAndTrackState(t *testing.T) {
	var h History
	var m ThreatMemory
	o := checkpointObservation(0)
	h.Enrich(&o)
	m.Enrich(&o, nil)
	state, err := CaptureObservationMemory(&h, &m, o.Identity)
	if err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*ObservationMemoryCheckpoint){
		func(s *ObservationMemoryCheckpoint) { s.Frames[0].Identity.Frame++ },
		func(s *ObservationMemoryCheckpoint) { s.NextTrack = 0 },
		func(s *ObservationMemoryCheckpoint) { s.Threats[0].Frame = s.Anchor.Frame + 1 },
		func(s *ObservationMemoryCheckpoint) { s.Version = "other" },
	} {
		s, _ := copyObservationMemory(state)
		mutate(&s)
		if _, _, err := RestoreObservationMemory(s, o.Identity); err == nil {
			t.Fatal("invalid observation memory admitted")
		}
	}
	native := o.Identity
	native.Actor++
	if _, _, err := RestoreObservationMemory(state, native); err == nil {
		t.Fatal("cross-actor restore admitted")
	}
}
