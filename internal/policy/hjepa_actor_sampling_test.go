package policy

import (
	"math"
	"q2coopbot/internal/quake"
	"reflect"
	"testing"
)

func hSamplingModel() HJEPAFile {
	f := hTestModel()
	f.Planner = "bc"
	f.PrimitiveSampling = &HJEPAActorSampling{Version: HJEPAActorDistribution, Seed: 23, LogStd: []float64{-2, -2, -6, -6}, Temperature: 1}
	return f
}

func TestHJEPAActorSamplingReplayAndLikelihood(t *testing.T) {
	p, err := LoadHJEPA(hWrite(t, hSamplingModel()))
	if err != nil {
		t.Fatal(err)
	}
	o := testObservation()
	seen := map[[4]float64]bool{}
	for frame := 0; frame < 32; frame++ {
		o.Identity.Frame++
		a, err := p.Decide(o)
		if err != nil {
			t.Fatal(err)
		}
		s := *p.LastSample()
		seen[s.Latent] = true
		b, err := p.Decide(o)
		if err != nil || a != b || !reflect.DeepEqual(s, *p.LastSample()) {
			t.Fatal("same-frame replay changed", err)
		}
		lp, err := p.PrimitiveLogProbability(o, s)
		if err != nil || math.Abs(lp-s.LogProbability) > 1e-12 {
			t.Fatal("likelihood mismatch", lp, s.LogProbability, err)
		}
		if a.Forward != math.Tanh(s.Latent[0]) || a.YawDelta != 180*math.Tanh(s.Latent[2]) || a.Attack != s.Attack || a.Vertical != []string{"release", "jump", "crouch"}[s.Vertical] {
			t.Fatal("raw proposal and receipt disagree")
		}
		p.ResetEpisode()
		if p.LastSample() != nil {
			t.Fatal("reset retained receipt")
		}
		c, err := p.Decide(o)
		if err != nil || c != a {
			t.Fatal("episode replay advanced RNG", err)
		}
	}
	if len(seen) != 32 {
		t.Fatal("counter sampling lacks frame separation")
	}
	o.Health = 0
	if _, err := p.Decide(o); err == nil || p.LastSample() != nil {
		t.Fatal("invalid observation retained sample")
	}
}

func TestHJEPAActorLatentDensityHasAnalyticScale(t *testing.T) {
	cfg := *hSamplingModel().PrimitiveSampling
	s := Sample{Version: HJEPAActorSampleVersion, SamplingSeed: cfg.Seed, Vertical: 0}
	raw := []float64{0, 0, 0, 0, -1, 1, 0, 0}
	lp, err := hActorLogProbability(raw, cfg, s)
	if err != nil {
		t.Fatal(err)
	}
	want := -2*math.Log(2*math.Pi) - cfg.LogStd[0] - cfg.LogStd[1] - cfg.LogStd[2] - cfg.LogStd[3] - math.Log1p(math.Exp(-1)) + 1 - math.Log(math.E+2)
	if math.Abs(lp-want) > 1e-12 {
		t.Fatal("latent-density convention changed", lp, want)
	}
	s.Latent[0] = math.NaN()
	if _, err := hActorLogProbability(raw, cfg, s); err == nil {
		t.Fatal("nonfinite sample admitted")
	}
	s.Latent[0] = 0
	s.Vertical = 3
	if _, err := hActorLogProbability(raw, cfg, s); err == nil {
		t.Fatal("invalid pose admitted")
	}
	s.Vertical = 0
	s.SamplingSeed++
	if _, err := hActorLogProbability(raw, cfg, s); err == nil {
		t.Fatal("wrong behavior seed admitted")
	}
}

func TestHJEPAActorCounterDistributionAndBounds(t *testing.T) {
	f := hSamplingModel()
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	raw := []float64{0, 0, 0, 0, -1, 1, 0, 0}
	o := testObservation()
	count := 4096
	sum, square, fire, release := 0., 0., 0., 0.
	for i := 0; i < count; i++ {
		o.Identity.Frame = i
		for stream := 0; stream < 10; stream++ {
			u := hActorUniform(23, o.Identity, stream)
			if !(u > 0 && u < 1) {
				t.Fatal("uniform endpoint")
			}
		}
		_, s, err := p.samplePrimitive(o, raw)
		if err != nil {
			t.Fatal(err)
		}
		z := s.Latent[0] / math.Exp(f.PrimitiveSampling.LogStd[0])
		sum += z
		square += z * z
		if s.Attack {
			fire++
		}
		if s.Vertical == 0 {
			release++
		}
	}
	if math.Abs(sum/float64(count)) > .06 || math.Abs(square/float64(count)-1) > .1 {
		t.Fatal("Gaussian scale drift", sum, square)
	}
	fireProbability, pose := hActorProbabilities(raw, 1)
	if math.Abs(fire/float64(count)-fireProbability) > .04 || math.Abs(release/float64(count)-pose[0]) > .04 {
		t.Fatal("categorical scale drift")
	}
}

func TestHJEPAActorJointWeaponLikelihood(t *testing.T) {
	for _, sampleWeapon := range []bool{false, true} {
		f := hSamplingModel()
		f.WeaponMode = "learned_masked_v1"
		f.WeaponHead = WeaponHeadVersion
		f.WeaponActor = []DenseLayer{hTestLayer(845, 12)}
		if sampleWeapon {
			seed := int64(31)
			f.WeaponSamplingSeed = &seed
		} else {
			f.WeaponActor[0].Bias[4] = 3
		}
		p, err := LoadHJEPA(hWrite(t, f))
		if err != nil {
			t.Fatal(err)
		}
		o := testObservation()
		items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 20}}
		age := 0
		o.Inventory = &items
		o.InventoryAgeFrames = &age
		a, err := p.Decide(o)
		if err != nil {
			t.Fatal(err)
		}
		s := *p.LastSample()
		if a.Weapon != weaponNames[s.Weapon] {
			t.Fatal("weapon receipt mismatch")
		}
		lp, err := p.PrimitiveLogProbability(o, s)
		if err != nil || math.Abs(lp-s.LogProbability) > 1e-12 {
			t.Fatal("joint probability mismatch", err)
		}
		base, err := hActorLogProbability([]float64{0, 0, 0, 0, -1, 1, 0, 0}, *f.PrimitiveSampling, s)
		if err != nil {
			t.Fatal(err)
		}
		if sampleWeapon {
			base -= math.Log(3)
		}
		if math.Abs(lp-base) > 1e-12 {
			t.Fatal("weapon factor missing/doubled", lp, base)
		}
		s.Weapon = 9
		if _, err := p.PrimitiveLogProbability(o, s); err == nil {
			t.Fatal("unavailable weapon sample admitted")
		}
	}
}

func TestHJEPAActorSamplingRejectsUntrackedTransformations(t *testing.T) {
	for _, mutate := range []func(*HJEPAFile){
		func(f *HJEPAFile) { f.Planner = "sequence" }, func(f *HJEPAFile) { f.Motor = []DenseLayer{hTestLayer(845, 8)} },
		func(f *HJEPAFile) { f.WeaponOptionFrames = 2 }, func(f *HJEPAFile) { f.PrimitiveSampling.Version = "unknown" },
		func(f *HJEPAFile) { f.PrimitiveSampling.Seed = -1 }, func(f *HJEPAFile) { f.PrimitiveSampling.LogStd = []float64{0, 0} },
		func(f *HJEPAFile) { f.PrimitiveSampling.LogStd[0] = -11 }, func(f *HJEPAFile) { f.PrimitiveSampling.Temperature = 0 },
	} {
		f := hSamplingModel()
		mutate(&f)
		if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
			t.Fatal("unsupported sampling contract admitted")
		}
	}
	f := hTestModel()
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	if _, err = p.Decide(testObservation()); err != nil || p.LastSample() != nil {
		t.Fatal("legacy model became stochastic", err)
	}
	if _, err = p.PrimitiveLogProbability(testObservation(), Sample{}); err == nil {
		t.Fatal("greedy policy admitted as sampled")
	}
}
