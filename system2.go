package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"
)

type StrategyDecision struct {
	Choice    string    `json:"choice"`
	Reason    string    `json:"reason"`
	At        time.Time `json:"at"`
	LatencyMS int64     `json:"latency_ms"`
}
type decisionResult struct {
	decision StrategyDecision
	err      error
}
type Strategist struct {
	model             string
	endpoint          string
	http              *http.Client
	pending           chan decisionResult
	busy              bool
	next              time.Time
	decisions, errors int
}

func NewStrategist(model string) *Strategist {
	return &Strategist{model: model, endpoint: "http://127.0.0.1:11434/api/generate", http: &http.Client{Timeout: 30 * time.Second}, pending: make(chan decisionResult, 1)}
}

type planState struct {
	Map                  string      `json:"map"`
	BotHP                int16       `json:"bot_hp"`
	BotAmmo              int16       `json:"bot_ammo"`
	BotWeapon            string      `json:"bot_weapon"`
	HumanHP              *int        `json:"human_hp"`
	HumanDistance        *int        `json:"human_distance"`
	Navigation           string      `json:"navigation"`
	Enemies              []planEnemy `json:"enemies,omitempty"`
	HealthPickupDistance *int        `json:"health_pickup_distance,omitempty"`
	CurrentPlan          string      `json:"current_plan"`
}
type planEnemy struct {
	ID            int   `json:"id"`
	DistanceBot   int   `json:"distance_bot"`
	DistanceHuman *int  `json:"distance_human"`
	ClearShot     *bool `json:"clear_shot"`
}

func compactPlanState(w World) planState {
	s := w.Snapshot
	state := planState{Map: w.Map, BotHP: s.Health, BotAmmo: s.Ammo, BotWeapon: s.Weapon, Navigation: w.Navigation, CurrentPlan: w.Goal}
	if s.Teammate != nil {
		v := int(math.Round(horizontal(s.Self, *s.Teammate)))
		state.HumanDistance = &v
	}
	for _, e := range s.Enemies {
		item := planEnemy{ID: e.ID, DistanceBot: int(math.Round(horizontal(s.Self, e.Origin))), ClearShot: e.ClearShot}
		if s.Teammate != nil {
			v := int(math.Round(horizontal(*s.Teammate, e.Origin)))
			item.DistanceHuman = &v
		}
		state.Enemies = append(state.Enemies, item)
	}
	sort.Slice(state.Enemies, func(i, j int) bool { return state.Enemies[i].DistanceBot < state.Enemies[j].DistanceBot })
	if len(state.Enemies) > 3 {
		state.Enemies = state.Enemies[:3]
	}
	for _, p := range s.Pickups {
		if !strings.Contains(p.Class, "health") {
			continue
		}
		v := int(math.Round(horizontal(s.Self, p.Origin)))
		if state.HealthPickupDistance == nil || v < *state.HealthPickupDistance {
			state.HealthPickupDistance = &v
		}
	}
	return state
}

const strategyRules = "You plan for a cooperative Quake II companion. Choose exactly one valid option from the supplied list. Interpret fields literally: a numeric human_distance proves the human is present. human_hp=null means UNKNOWN, never dead. A visible enemy has unknown health unless explicitly known; clear_shot is a static BSP line check, not a guarantee of a hit. current_plan is an old plan, not a reason to repeat it. Priorities: recover when bot_hp<45 and nearby health exists; protect the human from nearby enemies; regroup if human_distance>220; otherwise advance with the human. Never choose wait when human_distance is numeric. Return only JSON with choice and reason, where reason cites a field and its value in under 15 words."

func (s *Strategist) options(w World) []string {
	state := compactPlanState(w)
	if state.HumanDistance == nil {
		return []string{"wait"}
	}
	options := []string{"advance", "regroup", "cover"}
	if state.HealthPickupDistance != nil && *state.HealthPickupDistance <= 300 {
		options = append(options, "recover")
	}
	for _, e := range state.Enemies {
		if e.ClearShot != nil && *e.ClearShot {
			options = append(options, "engage")
			break
		}
		options = append(options, "reposition")
		break
	}
	return options
}
func (s *Strategist) poll(w World) (StrategyDecision, bool) {
	select {
	case result := <-s.pending:
		s.busy = false
		s.next = time.Now().Add(2 * time.Second)
		if result.err != nil {
			s.errors++
			log.Printf("system2 error: %v", result.err)
			return StrategyDecision{}, false
		}
		valid := false
		for _, choice := range s.options(w) {
			if result.decision.Choice == choice {
				valid = true
				break
			}
		}
		if !valid {
			s.errors++
			log.Printf("system2 invalid choice=%q allowed=%v", result.decision.Choice, s.options(w))
			return StrategyDecision{}, false
		}
		s.decisions++
		return result.decision, true
	default:
		return StrategyDecision{}, false
	}
}
func (s *Strategist) tick(w World) {
	if s.model == "" || s.busy || time.Now().Before(s.next) || w.Map == "" || w.Snapshot.Frame == 0 || w.Snapshot.Teammate == nil {
		return
	}
	s.busy = true
	state := compactPlanState(w)
	options := s.options(w)
	go func() {
		start := time.Now()
		stateJSON, _ := json.Marshal(state)
		optionJSON, _ := json.Marshal(options)
		prompt := strategyRules + "\nState: " + string(stateJSON) + "\nOptions: " + string(optionJSON)
		requestJSON, _ := json.Marshal(map[string]any{"model": s.model, "prompt": prompt, "stream": false, "think": false, "format": "json", "keep_alive": "10m", "options": map[string]any{"temperature": 0.1, "num_predict": 100, "num_ctx": 2048}})
		resp, e := s.http.Post(s.endpoint, "application/json", bytes.NewReader(requestJSON))
		if e != nil {
			s.pending <- decisionResult{err: e}
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			s.pending <- decisionResult{err: fmt.Errorf("ollama status %d", resp.StatusCode)}
			return
		}
		body, e := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if e != nil {
			s.pending <- decisionResult{err: e}
			return
		}
		var response struct {
			Response string `json:"response"`
		}
		if e = json.Unmarshal(body, &response); e != nil {
			s.pending <- decisionResult{err: e}
			return
		}
		var d StrategyDecision
		if e = json.Unmarshal([]byte(response.Response), &d); e != nil {
			s.pending <- decisionResult{err: fmt.Errorf("invalid Ollama response %q: %w", response.Response, e)}
			return
		}
		d.Choice = strings.ToLower(strings.TrimSpace(d.Choice))
		d.Reason = strings.TrimSpace(d.Reason)
		d.At = time.Now()
		d.LatencyMS = time.Since(start).Milliseconds()
		s.pending <- decisionResult{decision: d}
	}()
}
