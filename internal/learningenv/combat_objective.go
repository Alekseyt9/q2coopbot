package learningenv

import (
	"bufio"
	"fmt"
	"io"
	"q2coopbot/internal/policy"
	"regexp"
	"strconv"
	"strings"
)

type CombatObjective struct {
	Spawncount int    `json:"spawncount"`
	Frame      int    `json:"frame"`
	Map        string `json:"map"`
	GameFrame  int    `json:"game_frame"`
	Alive      int    `json:"alive"`
	Pending    int    `json:"pending"`
}

var objectivePattern = regexp.MustCompile(`^sv_test_objective spawncount=(-?\d+) server_frame=(\d+) g_test_objective version=1 map=([a-z][a-z0-9_]*) game_frame=(\d+) alive=(\d+) pending=(\d+)$`)

func ReadCombatObjectives(r io.Reader) ([]CombatObjective, error) {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), 1024*1024)
	ready := false
	seen := map[string]bool{}
	var result []CombatObjective
	for s.Scan() {
		line := strings.TrimSuffix(s.Text(), "\r")
		if line == "g_test_objective ready version=1" {
			ready = true
			continue
		}
		if !strings.HasPrefix(line, "sv_test_objective ") && !strings.HasPrefix(line, "g_test_objective ") {
			continue
		}
		m := objectivePattern.FindStringSubmatch(line)
		if m == nil {
			return nil, fmt.Errorf("invalid native objective event")
		}
		v := CombatObjective{Map: m[3]}
		for _, field := range []struct {
			i int
			p *int
		}{{1, &v.Spawncount}, {2, &v.Frame}, {4, &v.GameFrame}, {5, &v.Alive}, {6, &v.Pending}} {
			n, err := strconv.Atoi(m[field.i])
			if err != nil {
				return nil, err
			}
			*field.p = n
		}
		key := fmt.Sprintf("%d/%s/%d", v.Spawncount, v.Map, v.Frame)
		if seen[key] {
			return nil, fmt.Errorf("duplicate objective frame")
		}
		seen[key] = true
		result = append(result, v)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if !ready {
		return nil, fmt.Errorf("native objective telemetry not confirmed")
	}
	return result, nil
}

// Version2 requires independently parsed native state, not the receipt alone.
func VerifyGoalStopWithObjectiveForClasses(g GoalStop, release *CombatRelease, events []DamageEvent, observed policy.Observation, classes []string, mapName string, states []CombatObjective) error {
	if g.Version != "combat_goal_stop_v2" || g.NativeObjective == nil {
		return fmt.Errorf("native objective receipt missing")
	}
	want := *g.NativeObjective
	if want.Spawncount != g.Spawncount || want.Frame != g.ObservedFrame || want.Map != mapName || want.Alive != 0 || want.Pending != 0 {
		return fmt.Errorf("native objective incomplete or stale")
	}
	matched := 0
	for _, state := range states {
		if state == want {
			matched++
		}
	}
	if matched != 1 {
		return fmt.Errorf("native objective receipt not bound to original server state")
	}
	legacy := g
	legacy.Version = "combat_goal_stop_v1"
	legacy.NativeObjective = nil
	return VerifyGoalStopForClasses(legacy, release, events, observed, classes, mapName)
}
