package learningenv

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

const SynchronousVersion = "native_step_v1"

type NativeStep struct {
	Spawncount    int    `json:"spawncount"`
	BeginFrame    int    `json:"begin_frame"`
	EndFrame      int    `json:"end_frame"`
	Sequence      uint32 `json:"sequence"`
	Actor         int    `json:"actor"`
	DamageIndexes []int  `json:"damage_indexes"`
}
type CombatRelease struct {
	Spawncount int `json:"spawncount"`
	Frame      int `json:"frame"`
	GameFrame  int `json:"game_frame"`
	Seed       int `json:"seed"`
}
type NativeSteps struct {
	Steps   []NativeStep
	Release *CombatRelease
}

var stepPattern = regexp.MustCompile(`^sv_test_step version=1 phase=(begin|end) spawncount=(-?\d+) frame=(\d+) seq=(\d+) actor=(\d+)$`)
var releasePattern = regexp.MustCompile(`^sv_test_combat spawncount=(-?\d+) server_frame=(\d+) g_test_combat_start game_frame=(\d+) ready=1 seed=(\d+)$`)

// Preserve effect ordering, including immediate ClientThink damage emitted
// before the world frame counter increments. These are effects, not shot credit.
func ReadNativeSteps(r io.Reader, events []DamageEvent) (*NativeSteps, error) {
	return readNativeSteps(r, events, true)
}

// ReadNativeWindow parses a completed live pulse. A release is optional here;
// callers must retain and verify the actual earlier release independently.
func ReadNativeWindow(r io.Reader, events []DamageEvent) (*NativeSteps, error) {
	return readNativeSteps(r, events, false)
}

func readNativeSteps(r io.Reader, events []DamageEvent, requireRelease bool) (*NativeSteps, error) {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), 1024*1024)
	result := &NativeSteps{}
	var pending *NativeStep
	eventIndex := 0
	for s.Scan() {
		line := strings.TrimSuffix(s.Text(), "\r")
		if strings.HasPrefix(line, "sv_test_damage ") {
			if eventIndex >= len(events) {
				return nil, fmt.Errorf("native step damage count mismatch")
			}
			if pending != nil {
				e := events[eventIndex]
				if e.Spawncount != pending.Spawncount || e.Frame < pending.BeginFrame || e.Frame > pending.BeginFrame+1 {
					return nil, fmt.Errorf("native step effect outside world/frame")
				}
				pending.DamageIndexes = append(pending.DamageIndexes, eventIndex)
			}
			eventIndex++
			continue
		}
		if strings.HasPrefix(line, "sv_test_combat ") {
			m := releasePattern.FindStringSubmatch(line)
			if m == nil || result.Release != nil || pending == nil {
				return nil, fmt.Errorf("missing, duplicate or unsupported single-client release")
			}
			v := make([]int, 5)
			for i := 1; i <= 4; i++ {
				n, err := strconv.ParseInt(m[i], 10, 32)
				if err != nil {
					return nil, err
				}
				v[i] = int(n)
			}
			if v[1] != pending.Spawncount || v[2] != pending.BeginFrame+1 {
				return nil, fmt.Errorf("release outside native tick")
			}
			result.Release = &CombatRelease{Spawncount: v[1], Frame: v[2], GameFrame: v[3], Seed: v[4]}
			continue
		}
		if !strings.HasPrefix(line, "sv_test_step ") {
			continue
		}
		m := stepPattern.FindStringSubmatch(line)
		if m == nil {
			return nil, fmt.Errorf("native step rejected or malformed")
		}
		generation, err := strconv.ParseInt(m[2], 10, 32)
		if err != nil {
			return nil, err
		}
		frame, err := strconv.ParseInt(m[3], 10, 32)
		if err != nil {
			return nil, err
		}
		seq, err := strconv.ParseUint(m[4], 10, 31)
		if err != nil || seq == 0 {
			return nil, fmt.Errorf("invalid native step sequence")
		}
		actor, err := strconv.ParseInt(m[5], 10, 32)
		if err != nil || actor < 1 {
			return nil, fmt.Errorf("invalid native step actor")
		}
		if m[1] == "begin" {
			if pending != nil {
				return nil, fmt.Errorf("overlapping native steps")
			}
			if len(result.Steps) > 0 {
				previous := result.Steps[len(result.Steps)-1]
				if previous.Spawncount != int(generation) || previous.Actor != int(actor) || previous.EndFrame != int(frame) || previous.Sequence >= uint32(seq) {
					return nil, fmt.Errorf("native step discontinuity")
				}
			}
			pending = &NativeStep{Spawncount: int(generation), BeginFrame: int(frame), Actor: int(actor), Sequence: uint32(seq), DamageIndexes: []int{}}
		} else {
			if pending == nil || pending.Spawncount != int(generation) || pending.Actor != int(actor) || pending.Sequence != uint32(seq) || pending.BeginFrame+1 != int(frame) {
				return nil, fmt.Errorf("native end without exact one-frame step")
			}
			pending.EndFrame = int(frame)
			result.Steps = append(result.Steps, *pending)
			pending = nil
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if pending != nil || len(result.Steps) == 0 || requireRelease && result.Release == nil || eventIndex != len(events) {
		return nil, fmt.Errorf("incomplete native steps/release/effects: pending=%t steps=%d release=%t effects=%d/%d", pending != nil, len(result.Steps), result.Release != nil, eventIndex, len(events))
	}
	return result, nil
}
func (n *NativeSteps) Match(s *Step) (*NativeStep, error) {
	id := s.Observation.Identity
	for i := range n.Steps {
		p := &n.Steps[i]
		if p.Spawncount == id.Spawncount && p.Actor == id.Actor && p.Sequence == s.ClientSequence {
			if p.BeginFrame != id.Frame || s.Next != nil && (!sameWorld(s) || s.Next.Identity.Frame != p.EndFrame) {
				return nil, fmt.Errorf("observation not aligned with native step")
			}
			return p, nil
		}
	}
	return nil, fmt.Errorf("capture lacks native step")
}
