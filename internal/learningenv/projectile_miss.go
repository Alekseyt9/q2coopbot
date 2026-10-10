package learningenv

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

const MissRewardVersion = "combat_reward_v5"

type ProjectileMiss struct {
	Shot    uint32     `json:"shot"`
	Entity  int        `json:"entity"`
	Outcome string     `json:"outcome"`
	Launch  NativeStep `json:"launch"`
	End     NativeStep `json:"end"`
}

type ProjectileMissOutcome struct {
	Version string           `json:"version"`
	Misses  []ProjectileMiss `json:"misses"`
}

type projectileRecord struct {
	shot                  uint32
	entity, attacker, mod int
	launch, end           *NativeStep
	outcome               string
}

type ProjectileMissJoiner struct {
	records  []*projectileRecord
	eligible map[*projectileRecord]bool
	used     map[*projectileRecord]bool
}

var projectilePattern = regexp.MustCompile(`^sv_test_projectile spawncount=(-?\d+) server_frame=(\d+) g_test_projectile version=1 event=(spawn|end) map=(\w+) frame=(\d+) shot=(\d+) entity=(\d+) (.*)$`)

// ReadProjectileMisses pins launches and endings to their ordered native
// windows. Damage contacts, unresolved/freed shots and prep launches are never
// miss evidence. A launch's life is established later from client observations.
func ReadProjectileMisses(r io.Reader, damage []DamageEvent) (*ProjectileMissJoiner, error) {
	j := &ProjectileMissJoiner{eligible: map[*projectileRecord]bool{}, used: map[*projectileRecord]bool{}}
	shots := map[string]*projectileRecord{}
	maps := map[string]string{}
	ready := false
	var window *NativeStep
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if line == "g_test_projectile ready version=1" {
			ready = true
			continue
		}
		if strings.HasPrefix(line, "sv_test_step ") {
			m := stepPattern.FindStringSubmatch(line)
			if m == nil {
				return nil, fmt.Errorf("miss telemetry: invalid step")
			}
			n := make([]int, 4)
			for i := range n {
				v, e := strconv.ParseInt(m[i+2], 10, 32)
				if e != nil {
					return nil, e
				}
				n[i] = int(v)
			}
			if n[1] < 0 || n[2] <= 0 || n[3] <= 0 {
				return nil, fmt.Errorf("miss telemetry: invalid step identity")
			}
			if m[1] == "begin" {
				if window != nil {
					return nil, fmt.Errorf("miss telemetry: overlapping steps")
				}
				window = &NativeStep{Spawncount: n[0], BeginFrame: n[1], EndFrame: n[1] + 1, Sequence: uint32(n[2]), Actor: n[3]}
			} else {
				if window == nil || window.Spawncount != n[0] || window.EndFrame != n[1] || window.Sequence != uint32(n[2]) || window.Actor != n[3] {
					return nil, fmt.Errorf("miss telemetry: unmatched end")
				}
				window = nil
			}
			continue
		}
		if !strings.HasPrefix(line, "sv_test_projectile ") && !strings.HasPrefix(line, "g_test_projectile ") {
			continue
		}
		m := projectilePattern.FindStringSubmatch(line)
		if m == nil || !ready {
			return nil, fmt.Errorf("miss telemetry: malformed/unready projectile")
		}
		generation, e := strconv.ParseInt(m[1], 10, 32)
		if e != nil {
			return nil, e
		}
		frame, e := strconv.Atoi(m[2])
		if e != nil {
			return nil, e
		}
		shot, e := strconv.ParseUint(m[6], 10, 32)
		if e != nil || shot == 0 {
			return nil, fmt.Errorf("miss telemetry: invalid shot")
		}
		entity, e := strconv.Atoi(m[7])
		if e != nil || entity <= 0 {
			return nil, fmt.Errorf("miss telemetry: invalid entity")
		}
		if window != nil && (window.Spawncount != int(generation) || frame < window.BeginFrame || frame > window.EndFrame) {
			return nil, fmt.Errorf("miss telemetry: event outside ordered window")
		}
		fields := map[string]string{}
		for _, v := range strings.Fields(m[8]) {
			pair := strings.SplitN(v, "=", 2)
			if len(pair) != 2 || fields[pair[0]] != "" {
				return nil, fmt.Errorf("miss telemetry: invalid fields")
			}
			fields[pair[0]] = pair[1]
		}
		key := fmt.Sprintf("%d/%d", generation, shot)
		copyWindow := func() *NativeStep {
			if window == nil {
				return nil
			}
			v := *window
			return &v
		}
		if m[3] == "spawn" {
			if shots[key] != nil {
				return nil, fmt.Errorf("miss telemetry: duplicate launch")
			}
			attacker, e := strconv.Atoi(fields["attacker"])
			if e != nil || attacker <= 0 {
				return nil, fmt.Errorf("miss telemetry: invalid attacker")
			}
			mod, e := strconv.Atoi(fields["mod"])
			if e != nil || mod <= 0 {
				return nil, fmt.Errorf("miss telemetry: invalid mod")
			}
			x := &projectileRecord{shot: uint32(shot), entity: entity, attacker: attacker, mod: mod, launch: copyWindow(), outcome: "unresolved"}
			shots[key] = x
			maps[key] = m[4]
			j.records = append(j.records, x)
		} else {
			x := shots[key]
			if x == nil || x.outcome != "unresolved" || x.entity != entity || maps[key] != m[4] {
				return nil, fmt.Errorf("miss telemetry: unmatched/duplicate ending")
			}
			if _, e := strconv.Atoi(fields["target"]); e != nil {
				return nil, e
			}
			switch fields["outcome"] {
			case "damage", "geometry", "sky", "freed":
			default:
				return nil, fmt.Errorf("miss telemetry: unknown ending")
			}
			x.outcome = fields["outcome"]
			x.end = copyWindow()
			if x.launch != nil && x.end != nil && x.end.BeginFrame < x.launch.BeginFrame {
				return nil, fmt.Errorf("miss telemetry: ending before launch")
			}
		}
	}
	if e := scanner.Err(); e != nil {
		return nil, e
	}
	if !ready || window != nil {
		return nil, fmt.Errorf("miss telemetry: incomplete stream")
	}
	for _, d := range damage {
		if d.Shot == 0 {
			continue
		}
		x := shots[fmt.Sprintf("%d/%d", d.Spawncount, d.Shot)]
		if x == nil || x.entity != d.Inflictor || x.attacker != d.Attacker || x.mod != d.Mod || x.outcome == "geometry" || x.outcome == "sky" {
			return nil, fmt.Errorf("miss telemetry: contradictory damage contact")
		}
	}
	return j, nil
}

func sameNativeWindow(a, b *NativeStep) bool {
	return a != nil && b != nil && a.Spawncount == b.Spawncount && a.BeginFrame == b.BeginFrame && a.EndFrame == b.EndFrame && a.Sequence == b.Sequence && a.Actor == b.Actor
}

func (j *ProjectileMissJoiner) Enrich(s *Step, o *ServerOutcome) error {
	o.ProjectileMisses = &ProjectileMissOutcome{Version: "native_projectile_miss_v1", Misses: []ProjectileMiss{}}
	if !o.Available || s.Native == nil || s.Execution == nil || !s.Execution.Matched || !s.Execution.WindowExclusive || s.Execution.RecoveryCommands != 0 || s.Observation.Identity.Life != 1 || s.Observation.Health <= 0 {
		return nil
	}
	for _, x := range j.records {
		if x.mod != 1 || x.attacker != s.Observation.Identity.Actor {
			continue
		}
		if sameNativeWindow(x.launch, s.Native) {
			j.eligible[x] = true
		}
		if j.eligible[x] && sameNativeWindow(x.end, s.Native) && (x.outcome == "geometry" || x.outcome == "sky") {
			if j.used[x] {
				return fmt.Errorf("miss telemetry: repeated reward window")
			}
			j.used[x] = true
			o.ProjectileMisses.Misses = append(o.ProjectileMisses.Misses, ProjectileMiss{Shot: x.shot, Entity: x.entity, Outcome: x.outcome, Launch: *x.launch, End: *x.end})
		}
	}
	return nil
}
