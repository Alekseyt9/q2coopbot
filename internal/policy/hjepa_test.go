package policy

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"q2coopbot/internal/quake"
	"testing"
)

func TestHJEPAWeaponMaskAndCompatibility(t *testing.T) {
	f := hTestModel()
	f.Planner = "bc"
	old, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	f.WeaponMode = "learned_masked_v1"
	f.WeaponHead = WeaponHeadVersion
	l := hTestLayer(WeaponFeatureWidth, 12)
	l.Bias[9] = 1000
	l.Bias[3] = 100
	l.Bias[4] = 90
	l.Bias[1] = 80
	f.WeaponActor = []DenseLayer{l}
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	o := testObservation()
	age := 0
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}, {Name: "Super Shotgun", Count: 1}, {Name: "Shells", Count: 1}, {Name: "Machinegun", Count: 1}, {Name: "Bullets", Count: 2}}
	o.Inventory = &items
	o.InventoryAgeFrames = &age
	a, err := p.Decide(o)
	if err != nil {
		t.Fatal(err)
	}
	if a.Weapon != "Machinegun" {
		t.Fatalf("unavailable Railgun/SSG must be masked: %q", a.Weapon)
	}
	b, err := old.Decide(o)
	if err != nil {
		t.Fatal(err)
	}
	a.Weapon = ""
	if a != b {
		t.Fatal("weapon head changed primitive action")
	}
	items[2].Count = 2
	a, err = p.Decide(o)
	if err != nil || a.Weapon != "Super Shotgun" {
		t.Fatalf("SSG ammo threshold: %v %v", a, err)
	}
	age = 21
	a, err = p.Decide(o)
	if err != nil || a.Weapon != "" {
		t.Fatalf("stale inventory must keep: %v %v", a, err)
	}
	o.Inventory = nil
	a, err = p.Decide(o)
	if err != nil || a.Weapon != "" {
		t.Fatalf("unknown inventory must keep: %v %v", a, err)
	}
	for _, mutate := range []func(*HJEPAFile){func(f *HJEPAFile) { f.WeaponHead = "unknown" }, func(f *HJEPAFile) { f.WeaponActor = nil }, func(f *HJEPAFile) { f.WeaponMode = "keep_equipped_v1" }} {
		bad := f
		mutate(&bad)
		if _, err := LoadHJEPA(hWrite(t, bad)); err == nil {
			t.Fatal("invalid weapon metadata admitted")
		}
	}
}

func TestHJEPAWeaponSamplingReplay(t *testing.T) {
	f := hTestModel()
	f.Planner = "bc"
	f.WeaponMode = "learned_masked_v1"
	f.WeaponHead = WeaponHeadVersion
	l := hTestLayer(845, 12)
	for i := range l.Bias {
		l.Bias[i] = 1000
	}
	l.Bias[0] = 0
	l.Bias[1] = 0
	f.WeaponActor = []DenseLayer{l}
	seed := int64(17)
	f.WeaponSamplingSeed = &seed
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	o := testObservation()
	age := 0
	items := []quake.InventoryItem{{Name: "Blaster", Count: 1}}
	o.Inventory = &items
	o.InventoryAgeFrames = &age
	seen := map[string]bool{}
	for frame := 1; frame <= 64; frame++ {
		o.Identity.Frame = frame
		a, err := p.Decide(o)
		if err != nil {
			t.Fatal(err)
		}
		b, err := p.Decide(o)
		if err != nil || a != b {
			t.Fatal("categorical replay changed")
		}
		if a.Weapon != "" && a.Weapon != "Blaster" {
			t.Fatal("unavailable sampled weapon")
		}
		seen[a.Weapon] = true
	}
	if len(seen) != 2 {
		t.Fatal("head must sample learned distribution, not always argmax")
	}
}

func hTestLayer(in, out int) DenseLayer {
	l := DenseLayer{Weight: make([][]float64, out), Bias: make([]float64, out)}
	for i := range l.Weight {
		l.Weight[i] = make([]float64, in)
	}
	return l
}

func TestHJEPAInventoryDeclaration(t *testing.T) {
	f := hTestModel()
	for _, mode := range []string{"", "keep_equipped_v1"} {
		f.WeaponMode = mode
		p, err := LoadHJEPA(hWrite(t, f))
		if err != nil {
			t.Fatal(err)
		}
		if p.KeepsEquippedWeapon() != (mode != "") {
			t.Fatal("inventory admission must be explicit")
		}
		a, err := p.Decide(testObservation())
		if err != nil {
			t.Fatal(err)
		}
		if a.Weapon != "" {
			t.Fatal("keep-equipped must not invent a weapon selection")
		}
	}
	f.WeaponMode = "automatic"
	if _, err := LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("unknown weapon mode admitted")
	}
}
func hTestModel() HJEPAFile {
	f := HJEPAFile{Kind: HJEPAKind, Features: WeaponFeatureVersion, PriorWeight: .25}
	for i, s := range []int{1, 4, 16} {
		n := 2
		if i == 0 {
			n = WeaponFeatureWidth
		}
		enc := hTestLayer(n, 2)
		if i > 0 {
			enc.Weight[0][0] = 1
			enc.Weight[1][1] = 1
		}
		pred := hTestLayer(2+8*s, 2)
		for j := 0; j < s; j++ {
			pred.Weight[0][2+8*j] = 1 / float64(s)
		}
		utility := hTestLayer(2, 1)
		utility.Weight[0][0] = 2
		f.Levels = append(f.Levels, HJEPALevel{s, []DenseLayer{enc}, []DenseLayer{pred}, []DenseLayer{utility}})
	}
	a := hTestLayer(2, 8)
	a.Bias[4] = -1
	a.Bias[5] = 1
	f.Actor = []DenseLayer{a}
	return f
}
func hWrite(t *testing.T, f HJEPAFile) string {
	t.Helper()
	b, e := json.Marshal(f)
	if e != nil {
		t.Fatal(e)
	}
	p := filepath.Join(t.TempDir(), "model.json")
	if e = os.WriteFile(p, b, 0600); e != nil {
		t.Fatal(e)
	}
	return p
}
func TestHJEPAHierarchicalSubgoalAndIsolation(t *testing.T) {
	path := hWrite(t, hTestModel())
	p, e := LoadHJEPA(path)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = LoadProvider(path, false); e == nil {
		t.Fatal("asynchronous H-JEPA admitted")
	}
	if _, e = LoadProvider(path, true); e != nil {
		t.Fatal(e)
	}
	o := testObservation()
	a, e := p.Decide(o)
	if e != nil {
		t.Fatal(e)
	}
	if a.Forward != .5 || a.Side != 0 || a.Attack || a.Vertical != "release" || a.Identity != o.Identity || a.Weapon != "" {
		t.Fatalf("coarse plan not refined into primitive: %+v", a)
	}
	again, e := p.Decide(o)
	if e != nil || again != a {
		t.Fatal("repeated frame nondeterministic")
	}
	o.Identity.Frame++
	next, e := p.Decide(o)
	if e != nil || next.Identity != o.Identity {
		t.Fatal("identity not refreshed")
	}
	o.AgeMS = 301
	if _, e = p.Decide(o); e == nil {
		t.Fatal("stale observation accepted")
	}
	o.AgeMS = 0
	o.Velocity[0] = math.NaN()
	if _, e = p.Decide(o); e == nil {
		t.Fatal("NaN accepted")
	}
}
func TestHJEPARejectsMalformedNetworks(t *testing.T) {
	for _, mutate := range []func(*HJEPAFile){
		func(f *HJEPAFile) { f.Planner = "unknown" },
		func(f *HJEPAFile) { f.Levels[1].Stride = 3 },
		func(f *HJEPAFile) { f.Levels[0].Encoder[0].Weight[0] = []float64{0} },
		func(f *HJEPAFile) { f.Levels[2].Predictor[0].Bias = f.Levels[2].Predictor[0].Bias[:1] },
		func(f *HJEPAFile) { f.PriorWeight = 0 },
		func(f *HJEPAFile) { f.Actor[0].Weight[0][0] = 1e5 },
	} {
		f := hTestModel()
		mutate(&f)
		if _, e := LoadHJEPA(hWrite(t, f)); e == nil {
			t.Fatal("invalid model accepted")
		}
	}
}

func TestHJEPAPlannerAblations(t *testing.T) {
	for _, mode := range []string{"", "hierarchy", "flat", "bc"} {
		f := hTestModel()
		f.Planner = mode
		p, err := LoadHJEPA(hWrite(t, f))
		if err != nil {
			t.Fatal(err)
		}
		a, err := p.Decide(testObservation())
		if err != nil {
			t.Fatal(err)
		}
		want := .5
		if mode == "bc" {
			want = 0
		}
		if a.Forward != want {
			t.Fatalf("%s: got %v, want %v", mode, a.Forward, want)
		}
	}
}

func TestHJEPANonconstantPredictionAndRollout(t *testing.T) {
	f := hTestModel()
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	plan := make([][]float64, 16)
	for i := range plan {
		plan[i] = []float64{float64(i) / 16, 0, 0, 0, 0, 1, 0, 0}
	}
	states, err := p.hRoll(1, []float64{0, 0}, plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(states) != 4 || math.Abs(states[0][0]-.09375) > 1e-12 || math.Abs(states[3][0]-.84375) > 1e-12 {
		t.Fatalf("action windows were repeated or misaligned: %v", states)
	}
	if _, err = p.hRoll(1, []float64{0, 0}, plan[:3]); err == nil {
		t.Fatal("incomplete block accepted")
	}
	f.Planner = "sequence"
	p, err = LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	a, err := p.Decide(testObservation())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = Command(testObservation(), a, [3]int16{}); err != nil {
		t.Fatal(err)
	}
}

func TestHJEPADistilledMotorOverridesOnlyProposal(t *testing.T) {
	f := hTestModel()
	f.Planner = "bc"
	l := hTestLayer(WeaponFeatureWidth, 8)
	l.Bias[0] = .3
	l.Bias[4] = -1
	l.Bias[5] = 1
	f.Motor = []DenseLayer{l}
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	a, err := p.Decide(testObservation())
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(a.Forward-math.Tanh(.3)) > 1e-12 {
		t.Fatal("distilled proposal not applied")
	}
	x, _ := FeaturesForVersion(testObservation(), WeaponFeatureVersion)
	_, raw, _, err := p.Replay(x, []float64{0, 0, 0, 0, 0, 1, 0, 0})
	if err != nil || raw[0] != .3 {
		t.Fatal("replay motor differs from runtime")
	}
	f.Motor[0].Weight[0] = nil
	if _, err = LoadHJEPA(hWrite(t, f)); err == nil {
		t.Fatal("bad motor shape accepted")
	}
}

func TestHJEPASearchAxesPreserveUnselectedComponents(t *testing.T) {
	f := hTestModel()
	f.Planner = "sequence"
	f.PlanAxes = []int{0, 1}
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	prior := []float64{0, 0, .1, -.1, 1, 1, 0, 0}
	for _, a := range p.candidates(prior) {
		for i := 2; i < 8; i++ {
			if a[i] != prior[i] {
				t.Fatal("search changed preserved motor axis")
			}
		}
	}
	for _, axes := range [][]int{{0, 0}, {-1}, {5}} {
		f.PlanAxes = axes
		if _, err = LoadHJEPA(hWrite(t, f)); err == nil {
			t.Fatal("bad axis contract accepted")
		}
	}
}

func TestHJEPASmallStepsAndCrossHorizonAgreement(t *testing.T) {
	margin := 0.
	f := hTestModel()
	f.Planner = "sequence"
	f.PlanAxes = []int{0, 1}
	f.PlanStep = .1
	f.ConsistencyMargin = &margin
	p, err := LoadHJEPA(hWrite(t, f))
	if err != nil {
		t.Fatal(err)
	}
	prior := []float64{.95, 0, .2, .3, 1, 1, 0, 0}
	for _, a := range p.candidates(prior) {
		if math.Abs(a[0]-prior[0]) > .1000001 || math.Abs(a[1]-prior[1]) > .1000001 || a[2] != prior[2] || a[4] != prior[4] {
			t.Fatal("search escaped its motor trust region")
		}
	}
	// All three trained timescales agree on this synthetic positive transition.
	zs := [][]float64{{0, 0}, {0, 0}, {0, 0}}
	plan := make([][]float64, 16)
	for i := range plan {
		plan[i] = []float64{.1, 0, 0, 0, 0, 1, 0, 0}
	}
	zero := []float64{0, 0, 0, 0, 0, 1, 0, 0}
	a, err := p.admitPlan(zs, zero, plan)
	if err != nil || a[0] != .1 || !p.LastSearchAudit().Accepted {
		t.Fatal("consistent improvement rejected")
	}
	// One-step dynamics disagree; this cannot pass merely because the coarse
	// predictor is optimistic. The exact motor action must be returned.
	p.file.Levels[0].Predictor[0].Weight[0][2] = -1
	a, err = p.admitPlan(zs, zero, plan)
	if err != nil || a[0] != 0 || p.LastSearchAudit().Accepted {
		t.Fatal("disagreement did not preserve motor")
	}
	for _, step := range []float64{-.1, .6} {
		f.PlanStep = step
		if _, err = LoadHJEPA(hWrite(t, f)); err == nil {
			t.Fatal("invalid search step accepted")
		}
	}
}
