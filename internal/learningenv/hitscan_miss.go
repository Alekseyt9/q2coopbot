package learningenv

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
)

type HitscanMiss struct {
	Shot    uint32     `json:"shot"`
	Outcome string     `json:"outcome"`
	Window  NativeStep `json:"window"`
}
type HitscanMissOutcome struct {
	Version string        `json:"version"`
	Misses  []HitscanMiss `json:"misses"`
}
type hitscanRecord struct {
	shot              uint32
	generation, frame int
	world, outcome    string
	window            *NativeStep
}
type HitscanMissJoiner struct {
	records []*hitscanRecord
	used    map[*hitscanRecord]bool
}

var hitscanPattern = regexp.MustCompile(`^sv_test_hitscan spawncount=(-?\d+) server_frame=(\d+) g_test_hitscan version=1 event=fire (.*)$`)
var hitscanNamePattern = regexp.MustCompile(`^\w+$`)

// ReadHitscanMisses validates the completed ordered native stream. MG damage
// is synchronous and follows its fire record; an absence of damage alone is
// insufficient miss evidence. Damageable contacts remain unpenalized.
func ReadHitscanMisses(r io.Reader, damage []DamageEvent) (*HitscanMissJoiner, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	native, err := ReadNativeWindow(bytes.NewReader(b), damage)
	if err != nil {
		return nil, err
	}
	j := &HitscanMissJoiner{used: map[*hitscanRecord]bool{}}
	seen := map[string]bool{}
	var window *NativeStep
	var pending *hitscanRecord
	var fields map[string]string
	index, damageIndex := 0, 0
	s := bufio.NewScanner(bytes.NewReader(b))
	s.Buffer(make([]byte, 4096), 1024*1024)
	for s.Scan() {
		line := strings.TrimSuffix(s.Text(), "\r")
		if strings.HasPrefix(line, "sv_test_step ") {
			m := stepPattern.FindStringSubmatch(line) // Already strictly validated.
			if m[1] == "begin" {
				window = &native.Steps[index]
			} else {
				window = nil
				index++
			}
			pending = nil
			continue
		}
		if strings.HasPrefix(line, "sv_test_hitscan ") || strings.HasPrefix(line, "g_test_hitscan ") {
			m := hitscanPattern.FindStringSubmatch(line)
			if m == nil {
				return nil, fmt.Errorf("malformed hitscan telemetry")
			}
			generation, e := strconv.Atoi(m[1])
			if e != nil {
				return nil, e
			}
			frame, e := strconv.Atoi(m[2])
			if e != nil {
				return nil, e
			}
			f, e := validateHitscanFields(m[3])
			if e != nil {
				return nil, e
			}
			shot, _ := strconv.ParseUint(f["shot"], 10, 32)
			actor, _ := strconv.Atoi(f["actor"])
			key := fmt.Sprintf("%d/%d", generation, shot)
			if seen[key] {
				return nil, fmt.Errorf("duplicate hitscan shot")
			}
			seen[key] = true
			if window != nil && (generation != window.Spawncount || frame < window.BeginFrame || frame > window.EndFrame) {
				return nil, fmt.Errorf("hitscan outside native window")
			}
			pending = &hitscanRecord{shot: uint32(shot), world: f["map"], generation: generation, frame: frame}
			fields = f
			if window != nil && actor == window.Actor {
				pending.window = window
			}
			fraction, _ := strconv.ParseFloat(f["fraction"], 64)
			health, _ := strconv.Atoi(f["health_before"])
			switch {
			case f["sky"] == "1":
				pending.outcome = "sky"
			case fraction >= 1:
				pending.outcome = "no_contact"
			case f["damageable"] == "0":
				pending.outcome = "geometry"
			case health <= 0:
				pending.outcome = "corpse"
			}
			j.records = append(j.records, pending)
		}
		if strings.HasPrefix(line, "sv_test_damage ") {
			d := damage[damageIndex]
			damageIndex++
			if d.Mod != 4 {
				continue
			}
			if window != nil && d.Attacker == window.Actor && (pending == nil || pending.window != window) {
				return nil, fmt.Errorf("MG damage without matching fire")
			}
			if pending != nil && fmt.Sprint(d.Attacker) == fields["actor"] {
				if d.Spawncount != pending.generation || d.Frame != pending.frame || d.Map != pending.world || d.GameFrame != mustHitscanInt(fields["frame"]) ||
					d.Target != mustHitscanInt(fields["target"]) || d.HealthBefore != mustHitscanInt(fields["health_before"]) ||
					d.Inflictor != d.Attacker || (pending.outcome != "" && pending.outcome != "corpse") || fields["damageable"] == "0" {
					return nil, fmt.Errorf("contradictory hitscan damage")
				}
				pending = nil // A second damage record cannot reuse the shot.
			}
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return j, nil
}

func mustHitscanInt(s string) int { v, _ := strconv.Atoi(s); return v }

func validateHitscanFields(text string) (map[string]string, error) {
	f := map[string]string{}
	for _, token := range strings.Fields(text) {
		p := strings.SplitN(token, "=", 2)
		if len(p) != 2 || p[1] == "" || f[p[0]] != "" {
			return nil, fmt.Errorf("invalid hitscan fields")
		}
		f[p[0]] = p[1]
	}
	keys := strings.Fields("map frame shot actor mod start aim view recoil velocity spread nominal water muzzle_blocked fraction impact sky target target_class damageable health_before burst gunframe ammo_before damage")
	if len(f) != len(keys) {
		return nil, fmt.Errorf("missing/unknown hitscan fields")
	}
	for _, k := range keys {
		if f[k] == "" {
			return nil, fmt.Errorf("missing hitscan %s", k)
		}
	}
	if !hitscanNamePattern.MatchString(f["map"]) || !hitscanNamePattern.MatchString(f["target_class"]) {
		return nil, fmt.Errorf("invalid hitscan world/class")
	}
	for _, k := range strings.Fields("frame actor mod water muzzle_blocked sky target damageable health_before burst gunframe ammo_before damage") {
		if _, e := strconv.ParseInt(f[k], 10, 32); e != nil {
			return nil, e
		}
	}
	shot, e := strconv.ParseUint(f["shot"], 10, 32)
	if e != nil || shot == 0 || mustHitscanInt(f["actor"]) <= 0 || f["mod"] != "4" || mustHitscanInt(f["frame"]) < 0 || mustHitscanInt(f["target"]) < -1 || mustHitscanInt(f["ammo_before"]) <= 0 || mustHitscanInt(f["damage"]) <= 0 {
		return nil, fmt.Errorf("invalid hitscan identity")
	}
	for _, k := range []string{"sky", "water", "muzzle_blocked"} {
		if f[k] != "0" && f[k] != "1" {
			return nil, fmt.Errorf("invalid hitscan flag")
		}
	}
	if v := mustHitscanInt(f["damageable"]); v < 0 || v > 2 {
		return nil, fmt.Errorf("invalid native takedamage enum")
	}
	if mustHitscanInt(f["burst"]) < 0 || mustHitscanInt(f["burst"]) > 9 || (f["gunframe"] != "4" && f["gunframe"] != "5") {
		return nil, fmt.Errorf("invalid MG firing state")
	}
	vectors := map[string][]float64{}
	for _, k := range []string{"start", "aim", "view", "recoil", "velocity", "spread", "nominal", "impact", "fraction"} {
		count := 3
		if k == "spread" || k == "nominal" {
			count = 2
		}
		if k == "fraction" {
			count = 1
		}
		parts := strings.Split(f[k], ",")
		if len(parts) != count {
			return nil, fmt.Errorf("invalid hitscan vector")
		}
		for _, p := range parts {
			v, e := strconv.ParseFloat(p, 64)
			if e != nil || math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("nonfinite hitscan vector")
			}
			vectors[k] = append(vectors[k], v)
		}
	}
	if v := vectors["fraction"][0]; v < 0 || v > 1 {
		return nil, fmt.Errorf("invalid hitscan fraction")
	}
	pitch, yaw := (vectors["view"][0]+vectors["recoil"][0])*math.Pi/180, (vectors["view"][1]+vectors["recoil"][1])*math.Pi/180
	expected := []float64{math.Cos(pitch) * math.Cos(yaw), math.Cos(pitch) * math.Sin(yaw), -math.Sin(pitch)}
	for i, v := range vectors["aim"] {
		if math.Abs(v-expected[i]) > 5e-5 {
			return nil, fmt.Errorf("hitscan aim/recoil mismatch")
		}
	}
	for i, n := range vectors["nominal"] {
		factor := 1.0
		if f["water"] == "1" {
			factor = 2
		}
		if n < 0 || math.Abs(vectors["spread"][i]) > n*factor+.001 || f["muzzle_blocked"] == "1" && vectors["spread"][i] != 0 {
			return nil, fmt.Errorf("invalid hitscan spread")
		}
	}
	return f, nil
}

func (j *HitscanMissJoiner) Enrich(s *Step, o *ServerOutcome) error {
	o.HitscanMisses = &HitscanMissOutcome{Version: "native_hitscan_miss_v1", Misses: []HitscanMiss{}}
	if !o.Available || s.Native == nil || s.Execution == nil || !s.Execution.Matched || !s.Execution.WindowExclusive || s.Execution.RecoveryCommands != 0 || s.Observation.Identity.Life != 1 || s.Observation.Health <= 0 {
		return nil
	}
	for _, x := range j.records {
		if x.outcome == "" || !sameNativeWindow(x.window, s.Native) {
			continue
		}
		if x.world != s.Observation.Identity.Map || x.window.Actor != s.Observation.Identity.Actor {
			return fmt.Errorf("hitscan world mismatch")
		}
		if j.used[x] {
			return fmt.Errorf("repeated hitscan reward window")
		}
		j.used[x] = true
		o.HitscanMisses.Misses = append(o.HitscanMisses.Misses, HitscanMiss{Shot: x.shot, Outcome: x.outcome, Window: *x.window})
	}
	return nil
}
