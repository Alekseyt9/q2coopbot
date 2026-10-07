package policy

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

func weaponPPO(t *testing.T, deterministic bool) *PPO {
	t.Helper()
	f := writePPO(t, 40).file
	f.Features, f.WeaponHead, f.Deterministic = WeaponFeatureVersion, WeaponHeadVersion, deterministic
	for _, layers := range [][]DenseLayer{f.Actor, f.Value} {
		for i := range layers[0].Weight {
			layers[0].Weight[i] = make([]float64, WeaponFeatureWidth)
		}
	}
	for range weaponNames {
		f.Actor[2].Weight = append(f.Actor[2].Weight, make([]float64, len(f.Actor[1].Bias)))
		f.Actor[2].Bias = append(f.Actor[2].Bias, 0)
	}
	// A huge unavailable Railgun logit must have no effect on any head.
	f.Actor[2].Bias[8+9] = 1000
	path := filepath.Join(t.TempDir(), "weapon.json")
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	p, err := LoadPPO(path)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestWeaponHeadMaskedSamplesAndJointReview(t *testing.T) {
	p := weaponPPO(t, false)
	o := testObservation()
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 2}}
	age := 0
	o.Inventory, o.InventoryAgeFrames = &items, &age
	seen := map[int]bool{}
	for range 100 {
		a, err := p.Decide(o)
		if err != nil {
			t.Fatal(err)
		}
		s := *p.LastSample()
		seen[s.Weapon] = true
		if s.Weapon != 0 && s.Weapon != 1 && s.Weapon != 4 {
			t.Fatalf("sampled unavailable weapon: %d", s.Weapon)
		}
		b, lp, value, err := p.Review(o, s)
		if err != nil || b != a || lp != s.LogProbability || value != s.Value {
			t.Fatalf("joint replay: %v", err)
		}
		// Review the same latent draw through its legacy 8-head policy.
		legacy := *p
		legacy.file = p.file
		legacy.file.WeaponHead = ""
		legacy.file.Actor = append([]DenseLayer(nil), p.file.Actor...)
		legacy.file.Actor[2].Bias = p.file.Actor[2].Bias[:8]
		legacy.file.Actor[2].Weight = p.file.Actor[2].Weight[:8]
		legacy.actor = &MLP{file: MLPFile{Features: WeaponFeatureVersion, Layers: legacy.file.Actor}}
		copySample := s
		copySample.Weapon = 0
		_, oldLP, _, err := legacy.Review(o, copySample)
		if err != nil || math.Abs(lp-oldLP+math.Log(3)) > 1e-12 {
			t.Fatalf("weapon probability missing from joint logprob: %v", err)
		}
		s.Weapon = 9
		if _, _, _, err = p.Review(o, s); err == nil {
			t.Fatal("review accepted masked weapon")
		}
	}
	if len(seen) != 3 {
		t.Fatalf("weapon exploration missing: %v", seen)
	}
	age = 21
	a, err := p.Decide(o)
	if err != nil || a.Weapon != "" || p.LastSample().Weapon != 0 {
		t.Fatalf("stale switch: %v", err)
	}
	o.Inventory = nil
	a, err = p.Decide(o)
	if err != nil || a.Weapon != "" {
		t.Fatalf("unknown switch: %v", err)
	}
}

func TestWeaponHeadDeterministicAndStableLikelihood(t *testing.T) {
	p := weaponPPO(t, true)
	p.file.Actor[2].Bias[8+4] = 5
	o := testObservation()
	items := []quake.InventoryItem{{Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 1}}
	age := 0
	o.Inventory, o.InventoryAgeFrames = &items, &age
	a, err := p.Decide(o)
	if err != nil || a.Weapon != "Machinegun" || a.Vertical != "jump" {
		t.Fatalf("deterministic head: %+v %v", a, err)
	}
	logits := make([]float64, 12)
	logits[4] = 1000
	lp, err := weaponLogProbabilities(o, logits)
	if err != nil || lp[0] != -1000 || !math.IsInf(lp[9], -1) {
		t.Fatalf("stable masked logprob: %v %v", lp, err)
	}
}
