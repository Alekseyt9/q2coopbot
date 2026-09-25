package bot

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"
	"time"
)

type TacticalDecision struct {
	Action    string    `json:"action"`
	At        time.Time `json:"at"`
	LatencyMS int64     `json:"latency_ms"`
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
	return actions
}

var tacticLetter = regexp.MustCompile(`(?i)\b[A-D]\b`)

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
		if !allowed {
			t.errors++
			log.Printf("system1 stale action=%q", result.decision.Action)
			return TacticalDecision{}, false
		}
		t.decisions++
		return result.decision, true
	default:
		return TacticalDecision{}, false
	}
}
func (t *Tactician) tick(w World) {
	if t.model == "" || t.busy || time.Now().Before(t.next) || w.Map == "" || w.Snapshot.Frame == 0 || w.Snapshot.Teammate == nil {
		return
	}
	t.busy = true
	options := t.options(w)
	state := compactPlanState(w)
	go func() {
		start := time.Now()
		labels := make([]string, len(options))
		for i, option := range options {
			labels[i] = fmt.Sprintf("%c=%s", 'A'+i, option)
		}
		facts, _ := json.Marshal(state)
		prompt := "You are the fast tactical controller of a Quake II cooperative companion. Protect the human. Follow the current strategic plan. Attack only when clear_shot=true and the option is offered; network visibility alone is insufficient. Choose one offered option and output only its letter.\nState: " + string(facts) + "\nOptions: " + strings.Join(labels, ", ") + "\nAnswer:"
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
		t.pending <- tacticalResult{decision: TacticalDecision{Action: options[index], At: time.Now(), LatencyMS: time.Since(start).Milliseconds()}}
	}()
}
