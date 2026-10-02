package bot

import (
	"q2coopbot/internal/quake"
	"time"
)

// Fixed ASCII phrases work with the original Quake II console font.
// Debounce short-lived goals; rate limits use real time, even at 2x.
type planChat struct {
	candidate, mapName string
	sinceFrame         int
	lastSent           time.Time
	sent               map[string]time.Time
}

func planChatText(p *Planner) (string, string) {
	s := p.World.Snapshot
	if s.Health <= 0 || p.World.Navigation != "ready" && p.World.Navigation != "direct_clear" {
		return "", ""
	}
	switch p.World.Goal {
	case "regroup_after_respawn":
		return "regroup", "Vozvrashchayus k posledney tochke."
	case "search_last_seen", "probe_last_seen":
		return "search", "Ishchu tebya."
	case "follow_teammate":
		if s.Teammate != nil && quake.Horizontal(s.Self, *s.Teammate) > 256 {
			return "follow", "Dogonyayu tebya."
		}
	case "recover_health":
		if p.preparingForExit(s) {
			return "prepare", "Sobirayu resursy pered perehodom."
		}
		return "health", "Idu za aptechkoy."
	case "collect_item":
		if p.preparingSuppliesForExit(s) {
			return "prepare", "Sobirayu resursy pered perehodom."
		}
		return "items", "Idu za predmetom."
	case "reach_level_exit":
		return "exit", "Idu k vyhodu."
	}
	return "", ""
}

func (chat *planChat) command(p *Planner, now time.Time) string {
	key, text := planChatText(p)
	s := p.World.Snapshot
	if key != chat.candidate || s.Map != chat.mapName || s.Frame < chat.sinceFrame {
		chat.candidate, chat.mapName, chat.sinceFrame = key, s.Map, s.Frame
		return ""
	}
	if key == "" || s.Frame-chat.sinceFrame < 10 || !chat.lastSent.IsZero() && now.Sub(chat.lastSent) < 20*time.Second {
		return ""
	}
	if last := chat.sent[key]; !last.IsZero() && now.Sub(last) < 2*time.Minute {
		return ""
	}
	if chat.sent == nil {
		chat.sent = map[string]time.Time{}
	}
	chat.sent[key] = now
	chat.lastSent = now
	return "say \"" + text + "\""
}
