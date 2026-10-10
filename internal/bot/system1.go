package bot

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"q2coopbot/internal/quake"
	"regexp"
	"strings"
	"time"
)

type TacticalDecision struct {
	Action    string    `json:"action"`
	At        time.Time `json:"at"`
	LatencyMS int64     `json:"latency_ms"`
	Map       string    `json:"map,omitempty"`
	Frame     int       `json:"frame,omitempty"`
	Target    int       `json:"target,omitempty"`
	Source    string    `json:"source,omitempty"`
}

func (d TacticalDecision) current(s quake.Snapshot) bool {
	if s.Health <= 0 {
		return false
	}
	if d.Map != "" && (d.Map != s.Map || s.Frame < d.Frame || s.Frame-d.Frame > 20) {
		return false
	}
	if d.Target != 0 {
		for _, e := range s.Enemies {
			if e.ID == d.Target && e.ClearShot != nil && *e.ClearShot {
				return true
			}
		}
		return false
	}
	return true
}

type tacticalResult struct {
	decision TacticalDecision
	err      error
}
type Tactician struct {
	model, endpoint   string
	http              *http.Client
	pending           chan tacticalResult
	busy              bool
	next              time.Time
	decisions, errors int
}

func NewTactician(model string) *Tactician {
	return &Tactician{model: model, endpoint: "http://127.0.0.1:11434/api/generate", http: &http.Client{Timeout: 15 * time.Second}, pending: make(chan tacticalResult, 1)}
}
func (t *Tactician) options(w World) []string {
	s := w.Snapshot
	actions := []string{"follow"}
	if intent := combatIntent(w); intent != nil && intent.Action == "engage" {
		actions = nil // Route travel is not a tactic for an active campaign threat.
	}
	if w.Goal == "recover_health" {
		actions = append(actions, "recover")
	}
	for _, e := range s.Enemies {
		if e.ClearShot != nil && *e.ClearShot && (s.Ammo > 0 || strings.Contains(strings.ToLower(s.Weapon), "blast")) {
			actions = append(actions, "attack")
			break
		}
	}
	if s.Teammate != nil && w.Goal == "cover_teammate" {
		actions = append(actions, "hold")
	}
	if s.Teammate != nil && (w.Goal == "cover_teammate" || w.Goal == "follow_teammate") {
		if profile := combatSpacing(s); profile != nil && profile.NeedSpace && s.OnGround && quake.Distance(s.Self, *s.Teammate) <= combatLeash(profile) {
			actions = append(actions, "retreat")
		}
	}
	if w.Campaign != nil && w.Goal == "reach_level_exit" {
		if profile := combatSpacing(s); profile != nil && profile.NeedSpace && s.OnGround {
			actions = append(actions, "retreat")
		}
	}
	if plannedCover(w) != nil {
		actions = append(actions, "cover")
	}
	if _, _, ok := circleStep(w, 0); ok {
		actions = append(actions, "circle")
	}
	return actions
}

var tacticLetter = regexp.MustCompile(`(?i)\b[A-F]\b`)

func (t *Tactician) poll(w World) (TacticalDecision, bool) {
	select {
	case result := <-t.pending:
		t.busy = false
		t.next = time.Now().Add(time.Second)
		if result.err != nil {
			t.errors++
			log.Printf("system1 error: %v", result.err)
			return TacticalDecision{}, false
		}
		allowed := false
		for _, option := range t.options(w) {
			if option == result.decision.Action {
				allowed = true
				break
			}
		}
		if !allowed || !result.decision.current(w.Snapshot) {
			t.errors++
			log.Printf("system1 stale action=%q latency=%dms frame=%d current=%d", result.decision.Action, result.decision.LatencyMS, result.decision.Frame, w.Snapshot.Frame)
			return TacticalDecision{}, false
		}
		t.decisions++
		return result.decision, true
	default:
		return TacticalDecision{}, false
	}
}
func (t *Tactician) tick(w World) {
	if t.model == "" || t.busy || time.Now().Before(t.next) || w.Map == "" || w.Snapshot.Frame == 0 || w.Snapshot.Health <= 0 || !w.Snapshot.OnGround || math.Abs(w.Snapshot.SelfVelocity[2]) > 1 || (w.Snapshot.Teammate == nil && (w.Campaign == nil || combatSpacing(w.Snapshot) == nil)) {
		return
	}
	if w.Geometry != nil {
		if drop, ok := w.Geometry.GroundDrop(w.Snapshot.Self, 4); ok && drop > 1 {
			return
		}
	}
	t.busy = true
	options := t.options(w)
	state := struct {
		CoverAvailable bool           `json:"cover_available"`
		Health         int16          `json:"health"`
		Goal           string         `json:"goal"`
		Combat         *CombatSpacing `json:"combat"`
		Intent         *CombatIntent  `json:"intent"`
	}{plannedCover(w) != nil, w.Snapshot.Health, w.Goal, combatSpacing(w.Snapshot), combatIntent(w)}
	log.Printf("system1 request frame=%d cover_available=%t options=%v", w.Snapshot.Frame, state.CoverAvailable, options)
	go func() {
		start := time.Now()
		labels := make([]string, len(options))
		for i, option := range options {
			description := option
			switch option {
			case "attack":
				description = "stand still and attack in the open"
			case "cover":
				description = "hide behind wall, peek to fire, return"
			case "retreat":
				description = "back away while firing"
			case "circle":
				description = "move sideways around enemy while firing, keep distance"
			case "follow":
				description = "follow route"
			}
			labels[i] = fmt.Sprintf("%c=%s", 'A'+i, description)
		}
		facts := fmt.Sprintf("Quake II health=%d. Goal=%s. ", state.Health, state.Goal)
		if state.Intent != nil {
			facts += fmt.Sprintf("Campaign intent=%s (%s). ", state.Intent.Action, state.Intent.Reason)
		}
		if c := state.Combat; c != nil {
			facts += fmt.Sprintf("Enemy %s, weapon %s, distance%.0f, minimum%.0f, need_space=%t, threats=%d. ", c.Enemy, c.Weapon, c.Distance, c.Minimum, c.NeedSpace, c.VisibleThreats)
		}
		prompt := facts + fmt.Sprintf("Verified wall cover available=%t. Choose a tactic to reduce damage; back away if too close. ", state.CoverAvailable) + strings.Join(labels, ". ") + ". Answer one letter:"
		payload, _ := json.Marshal(map[string]any{"model": t.model, "prompt": prompt, "stream": false, "think": false, "keep_alive": "10m", "options": map[string]any{"temperature": 0, "num_predict": 1, "num_ctx": 1024}})
		resp, e := t.http.Post(t.endpoint, "application/json", bytes.NewReader(payload))
		if e != nil {
			t.pending <- tacticalResult{err: e}
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			t.pending <- tacticalResult{err: fmt.Errorf("ollama status %d", resp.StatusCode)}
			return
		}
		body, e := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if e != nil {
			t.pending <- tacticalResult{err: e}
			return
		}
		var result struct {
			Response string `json:"response"`
		}
		if e = json.Unmarshal(body, &result); e != nil {
			t.pending <- tacticalResult{err: e}
			return
		}
		letter := tacticLetter.FindString(strings.TrimSpace(result.Response))
		if letter == "" {
			t.pending <- tacticalResult{err: fmt.Errorf("invalid tactical answer %q", result.Response)}
			return
		}
		index := int(strings.ToUpper(letter)[0] - 'A')
		if index < 0 || index >= len(options) {
			t.pending <- tacticalResult{err: fmt.Errorf("tactical index out of range %q", letter)}
			return
		}
		d := TacticalDecision{Action: options[index], At: time.Now(), LatencyMS: time.Since(start).Milliseconds(), Map: w.Map, Frame: w.Snapshot.Frame, Source: "live"}
		if (d.Action == "attack" || d.Action == "retreat" || d.Action == "cover" || d.Action == "circle") && state.Combat != nil {
			d.Target = state.Combat.Target
		}
		t.pending <- tacticalResult{decision: d}
	}()
}
