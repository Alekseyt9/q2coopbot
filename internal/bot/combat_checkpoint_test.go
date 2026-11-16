package bot

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func combatCheckpointModel(t *testing.T) string {
	t.Helper()
	layer := func(in, out int) policy.DenseLayer {
		l := policy.DenseLayer{Weight: make([][]float64, out), Bias: make([]float64, out)}
		for i := range l.Weight {
			l.Weight[i] = make([]float64, in)
		}
		return l
	}
	f := policy.HJEPAFile{Kind: policy.HJEPAKind, Features: policy.HJEPAContextFeatureVersion, Planner: "bc", PriorWeight: .25,
		PrimitiveSampling: &policy.HJEPAActorSampling{Version: policy.HJEPAActorDistribution, Seed: 1729, LogStd: []float64{-2, -2, -6, -6}, Temperature: 1}}
	for i, stride := range []int{1, 4, 16} {
		in := 2
		if i == 0 {
			in = policy.HJEPAWeaponFeatureWidth
		}
		f.Levels = append(f.Levels, policy.HJEPALevel{Stride: stride, Encoder: []policy.DenseLayer{layer(in, 2)}, Predictor: []policy.DenseLayer{layer(2+8*stride, 2)}, Utility: []policy.DenseLayer{layer(2, 1)}})
	}
	f.Actor = []policy.DenseLayer{layer(2, 8)}
	// Make the actor depend on remembered context, not just constant biases.
	f.Levels[0].Encoder[0].Weight[0][0] = .3
	f.Actor[0].Weight[0][0] = .7
	f.WeaponMode, f.WeaponHead, f.WeaponFeatures = "learned_masked_v1", policy.WeaponHeadVersion, policy.HJEPAWeaponFeatureVersion
	f.WeaponActor = []policy.DenseLayer{layer(policy.HJEPAWeaponFeatureWidth, 12)}
	seed := int64(93)
	f.WeaponSamplingSeed = &seed
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "actor.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func checkpointTestClient(t *testing.T, path string, connection, generation int) *Client {
	t.Helper()
	p, err := policy.LoadHJEPA(path)
	if err != nil {
		t.Fatal(err)
	}
	return &Client{connection: connection, spawncount: generation, decoder: &quake.Decoder{PlayerNumber: 1}, planner: &Planner{},
		testSynchronous: true, testCombatGo: true,
		combatControl: combatControl{mode: "learned", provider: p, executionContract: policy.StockWeaponsFireCombatExecution}}
}

func checkpointTestSnapshot(frame, step int) quake.Snapshot {
	clear := true
	s := quake.Snapshot{Map: "base1", Frame: frame, Health: 100, OnGround: step%3 != 0,
		Self: quake.Vec3{float64(step), 10, 24}, SelfVelocity: quake.Vec3{10, 0, 0},
		Weapon: "models/weapons/v_handgr/tris.md2", GunFrame: 11, Ammo: 5, InventoryKnown: true,
		Inventory:   []quake.InventoryItem{{Name: "Grenades", Count: 5}, {Name: "Blaster", Count: 1}},
		Projectiles: []quake.Object{{ID: 500, Class: "grenade", Origin: quake.Vec3{40 + float64(step), 0, 24}}}}
	if step < 6 || step >= 12 {
		s.Enemies = []quake.Object{{ID: 7, Class: "monster_soldier", Origin: quake.Vec3{140 + float64(step), 0, 24}, ClearShot: &clear}}
	}
	return s
}

func TestCombatCheckpointPreDecisionContinuation(t *testing.T) {
	path := combatCheckpointModel(t)
	source := checkpointTestClient(t, path, 2, 3)
	observe := func(c *Client, frame, step int) policy.Observation {
		c.planner.World.Snapshot = checkpointTestSnapshot(frame, step)
		c.planner.World.Updated = time.Now()
		o := c.combatObservation(time.Now())
		c.combatControl.engagement.observe(o, nil)
		return o
	}
	for step := 0; step < 8; step++ {
		o := observe(source, 170+step, step)
		if _, err := source.combatControl.provider.Decide(o); err != nil {
			t.Fatal(err)
		}
		source.previous = quake.UserCmd{Buttons: 1, Forward: int16(step), Msec: 100}
		if step == 3 {
			source.combatControl.weaponRequests.record(o.Identity, "Grenades")
		}
		if step == 7 {
			source.combatControl.previousTarget = &policy.TargetIntent{Identity: o.Identity, Entity: 7, Track: 1}
		}
	}
	root := source.combatControl.last
	root.Frame++
	state, err := source.combatControl.captureCombatCheckpoint(root, source.previous)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	var saved combatCheckpoint
	if err = json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	loaded := checkpointTestClient(t, path, 4, 55)
	native := root
	native.Frame, native.Connection, native.Spawncount = 1, 4, 55
	loaded.previous, err = loaded.combatControl.restoreCombatCheckpoint(saved, native)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.combatControl.last.Frame != 0 || loaded.combatControl.observation != nil || loaded.previous != source.previous {
		t.Fatal("invalid pre-decision boundary")
	}
	loadedRootStep := 8
	for step := 8; step < 40; step++ {
		a, b := observe(source, 170+step, step), observe(loaded, 1+step-loadedRootStep, step)
		fa, err := policy.HJEPAFeatures(a, policy.HJEPAContextFeatureVersion)
		if err != nil {
			t.Fatal(err)
		}
		fb, err := policy.HJEPAFeatures(b, policy.HJEPAContextFeatureVersion)
		if err != nil || !reflect.DeepEqual(fa, fb) {
			t.Fatalf("features diverged at %d: %v", step, err)
		}
		if !reflect.DeepEqual(a.RememberedThreats, b.RememberedThreats) || !reflect.DeepEqual(a.PreviousWeaponRequestAgeFrames, b.PreviousWeaponRequestAgeFrames) {
			t.Fatalf("memory diverged at %d", step)
		}
		want, err := source.combatControl.provider.Decide(a)
		if err != nil {
			t.Fatal(err)
		}
		got, err := loaded.combatControl.provider.Decide(b)
		want.Identity = got.Identity
		if err != nil || want != got {
			t.Fatalf("action diverged at %d: %v", step, err)
		}
		sp, lp := source.combatControl.provider.(*policy.HJEPA), loaded.combatControl.provider.(*policy.HJEPA)
		if !reflect.DeepEqual(sp.LastSample(), lp.LastSample()) {
			t.Fatalf("sample diverged at %d", step)
		}
		source.previous = quake.UserCmd{Buttons: byte(step % 2), Forward: int16(step), Msec: 100}
		loaded.previous = source.previous
		source.combatControl.previousTarget, loaded.combatControl.previousTarget = nil, nil
		if step == 16 {
			// A second save must retain the original sampling clock, while
			// memory and request ages rebase from the first loaded session.
			nextRoot := loaded.combatControl.last
			nextRoot.Frame++
			nested, err := loaded.combatControl.captureCombatCheckpoint(nextRoot, loaded.previous)
			if err != nil {
				t.Fatal(err)
			}
			third := checkpointTestClient(t, path, 8, 88)
			thirdRoot := nextRoot
			thirdRoot.Frame, thirdRoot.Connection, thirdRoot.Spawncount = 1, 8, 88
			third.previous, err = third.combatControl.restoreCombatCheckpoint(nested, thirdRoot)
			if err != nil {
				t.Fatal(err)
			}
			loaded, loadedRootStep = third, step+1
		}
	}
	after, _ := json.Marshal(state)
	if string(after) != string(raw) {
		t.Fatal("continued source mutated saved state")
	}
}

func TestCombatCheckpointRejectsMissingPrefixAndConfiguration(t *testing.T) {
	path := combatCheckpointModel(t)
	c := checkpointTestClient(t, path, 2, 3)
	for step := 0; step < 5; step++ {
		c.planner.World.Snapshot = checkpointTestSnapshot(170+step, step)
		c.planner.World.Updated = time.Now()
		o := c.combatObservation(time.Now())
		c.combatControl.engagement.observe(o, nil)
	}
	root := c.combatControl.last
	if _, err := c.combatControl.captureCombatCheckpoint(root, c.previous); err == nil {
		t.Fatal("post-observation root admitted")
	}
	root.Frame++
	s, err := c.combatControl.captureCombatCheckpoint(root, c.previous)
	if err != nil {
		t.Fatal(err)
	}
	native := root
	native.Frame = 1
	native.Connection = 4
	native.Spawncount = 55
	for _, mutate := range []func(*combatCheckpoint){
		func(s *combatCheckpoint) { s.Root.Frame++ },
		func(s *combatCheckpoint) { s.LastHealth-- },
		func(s *combatCheckpoint) { s.Engagement.Identity.Frame-- },
		func(s *combatCheckpoint) {
			s.WeaponRequest = &combatCheckpointWeaponRequest{Identity: s.Root, Name: "Grenades"}
		},
		func(s *combatCheckpoint) {
			s.PreviousTarget = &policy.TargetIntent{Identity: s.Root, Entity: 7, Track: 1}
		},
	} {
		data, _ := json.Marshal(s)
		var bad combatCheckpoint
		_ = json.Unmarshal(data, &bad)
		mutate(&bad)
		fresh := checkpointTestClient(t, path, 4, 55)
		if _, err := fresh.combatControl.restoreCombatCheckpoint(bad, native); err == nil {
			t.Fatal("corrupt state admitted")
		}
		if fresh.combatControl.last != (policy.Identity{}) {
			t.Fatal("failed restore mutated controller")
		}
	}
	fresh := checkpointTestClient(t, path, 4, 55)
	fresh.combatControl.checkedWallSlide = true
	if _, err := fresh.combatControl.restoreCombatCheckpoint(s, native); err == nil {
		t.Fatal("changed execution contract admitted")
	}
	c.combatControl.forcedDuckProbeFrame = -1
	if _, err := c.combatControl.captureCombatCheckpoint(root, c.previous); err == nil {
		t.Fatal("unserialized diagnostic admitted")
	}
}
