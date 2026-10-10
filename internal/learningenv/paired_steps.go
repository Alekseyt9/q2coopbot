package learningenv

import (
	"bufio"
	"fmt"
	"io"
	"q2coopbot/internal/harness"
	"regexp"
	"strconv"
	"strings"
)

const PairedStepVersion = "native_paired_step_v1"

type NativePair struct {
	Steps    [2]NativeStep             `json:"steps"`
	Commands [2]harness.AppliedCommand `json:"commands"`
}
type NativePairs struct {
	Version string         `json:"version"`
	Pairs   []NativePair   `json:"pairs"`
	Release *CombatRelease `json:"release"`
}

var pairPhasePattern = regexp.MustCompile(`^sv_test_pair_step version=1 phase=(queued|end) spawncount=(\d+) frame=(\d+) role=([01]) seq=(\d+) actor=(\d+)$`)
var pairCommandPattern = regexp.MustCompile(`^sv_test_pair_cmd version=1 spawncount=(\d+) frame=(\d+) role=([01]) seq=(\d+) actor=(\d+) (.*)$`)
var pairReleasePattern = regexp.MustCompile(`^sv_test_combat spawncount=(\d+) server_frame=(\d+) g_test_combat_start game_frame=(\d+) ready=2 seed=(\d+)$`)

// ReadPairedNativeSteps verifies both participants before exposing any step.
// Damage indexes denote the shared effect window; downstream rewards must
// still attribute effects to each actor, rather than crediting both players.
// This is protocol evidence only, not a reset or training-eligibility seal.
func ReadPairedNativeSteps(r io.Reader, events []DamageEvent) (*NativePairs, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	result := &NativePairs{Version: PairedStepVersion}
	var pending NativePair
	queued, applied, ended, eventIndex := 0, 0, 0, 0
	number := func(s string) (int, error) { n, e := strconv.ParseInt(s, 10, 32); return int(n), e }
	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if strings.HasPrefix(line, "sv_test_damage ") {
			if eventIndex >= len(events) {
				return nil, fmt.Errorf("paired damage count mismatch")
			}
			if applied > 0 {
				e := events[eventIndex]
				p := pending.Steps[0]
				if e.Spawncount != p.Spawncount || e.Frame < p.BeginFrame || e.Frame > p.BeginFrame+1 {
					return nil, fmt.Errorf("paired effect outside tick")
				}
				for role := 0; role < 2; role++ {
					pending.Steps[role].DamageIndexes = append(pending.Steps[role].DamageIndexes, eventIndex)
				}
			}
			eventIndex++
			continue
		}
		if strings.HasPrefix(line, "sv_test_combat ") {
			m := pairReleasePattern.FindStringSubmatch(line)
			if m == nil || result.Release != nil || applied != 2 || ended != 0 {
				return nil, fmt.Errorf("invalid paired release")
			}
			v := make([]int, 4)
			for i := range v {
				var err error
				v[i], err = number(m[i+1])
				if err != nil {
					return nil, err
				}
			}
			if v[0] != pending.Steps[0].Spawncount || v[1] != pending.Steps[0].BeginFrame+1 {
				return nil, fmt.Errorf("paired release outside tick")
			}
			result.Release = &CombatRelease{Spawncount: v[0], Frame: v[1], GameFrame: v[2], Seed: v[3]}
			continue
		}
		if strings.HasPrefix(line, "sv_test_pair_cmd ") {
			m := pairCommandPattern.FindStringSubmatch(line)
			if m == nil || queued != 3 || applied >= 2 || ended != 0 {
				return nil, fmt.Errorf("paired command without both queued actors")
			}
			role, err := number(m[3])
			if err != nil {
				return nil, err
			}
			if role != applied {
				return nil, fmt.Errorf("paired command application order changed")
			}
			actor, err := number(m[5])
			if err != nil {
				return nil, err
			}
			c, err := harness.ParseAppliedCommand(fmt.Sprintf("sv_test_applied_cmd spawncount=%s frame=%s seq=%s kind=new %s", m[1], m[2], m[4], m[6]), 1)
			if err != nil {
				return nil, err
			}
			p := pending.Steps[role]
			if c.Generation != p.Spawncount || c.Frame != p.BeginFrame || c.Sequence != p.Sequence || actor != p.Actor {
				return nil, fmt.Errorf("paired command identity mismatch")
			}
			pending.Commands[role] = c
			applied++
			continue
		}
		if !strings.HasPrefix(line, "sv_test_pair_step ") {
			continue
		}
		m := pairPhasePattern.FindStringSubmatch(line)
		if m == nil {
			return nil, fmt.Errorf("paired step rejected or malformed")
		}
		v := make([]int, 5)
		for i := range v {
			var err error
			v[i], err = number(m[i+2])
			if err != nil {
				return nil, err
			}
		}
		generation, frame, role, seq, actor := v[0], v[1], v[2], v[3], v[4]
		if seq <= 0 || actor <= 0 {
			return nil, fmt.Errorf("invalid paired actor/sequence")
		}
		if m[1] == "queued" {
			if queued&(1<<role) != 0 || applied != 0 || ended != 0 {
				return nil, fmt.Errorf("duplicate or overlapping paired queue")
			}
			if queued != 0 {
				p := pending.Steps[1-role]
				if p.Spawncount != generation || p.BeginFrame != frame || p.Actor == actor {
					return nil, fmt.Errorf("pair actors do not share frame/world")
				}
			}
			if len(result.Pairs) > 0 {
				previous := result.Pairs[len(result.Pairs)-1].Steps[role]
				if previous.Spawncount != generation || previous.EndFrame != frame || previous.Actor != actor || previous.Sequence >= uint32(seq) {
					return nil, fmt.Errorf("paired step discontinuity")
				}
			}
			pending.Steps[role] = NativeStep{Spawncount: generation, BeginFrame: frame, Actor: actor, Sequence: uint32(seq)}
			queued |= 1 << role
		} else {
			p := &pending.Steps[role]
			if queued != 3 || applied != 2 || ended != role || p.Spawncount != generation || p.BeginFrame+1 != frame || p.Actor != actor || p.Sequence != uint32(seq) {
				return nil, fmt.Errorf("paired closure lacks exact next tick")
			}
			p.EndFrame = frame
			ended++
			if ended == 2 {
				result.Pairs = append(result.Pairs, pending)
				pending = NativePair{}
				queued, applied, ended = 0, 0, 0
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if queued != 0 || applied != 0 || ended != 0 || len(result.Pairs) == 0 || result.Release == nil || eventIndex != len(events) {
		return nil, fmt.Errorf("incomplete paired steps/release/effects")
	}
	return result, nil
}
