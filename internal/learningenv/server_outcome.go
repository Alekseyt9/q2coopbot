package learningenv

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

const ServerOutcomeVersion = "server_damage_window_v1"

type DamageEvent struct {
	Map           string `json:"map"`
	Spawncount    int    `json:"spawncount"`
	Frame         int    `json:"frame"`
	GameFrame     int    `json:"game_frame"`
	Attacker      int    `json:"attacker"`
	Target        int    `json:"target"`
	Inflictor     int    `json:"inflictor"`
	Mod           int    `json:"mod"`
	HealthBefore  int    `json:"health_before"`
	HealthAfter   int    `json:"health_after"`
	Take          int    `json:"take"`
	Armor         int    `json:"armor"`
	Power         int    `json:"power"`
	Protection    int    `json:"protection"`
	TargetClass   string `json:"target_class"`
	AttackerClass string `json:"attacker_class"`
	Shot          uint32 `json:"shot"`
}

var damagePattern = regexp.MustCompile(`^sv_test_damage spawncount=(-?\d+) server_frame=(\d+) g_test_damage version=1 map=(\w+) frame=(\d+) attacker=(\d+) target=(\d+) inflictor=(\d+) mod=(\d+) health_before=(-?\d+) health_after=(-?\d+) take=(\d+) armor=(\d+) power=(\d+) protection=(\d+) target_class=(\w+) attacker_class=(\w+)(?: shot=(\d+))?$`)

// ReadDamageEvents mirrors the native v1 telemetry contract. No permissive
// partial parsing: a missing ready marker or malformed event invalidates it.
func ReadDamageEvents(r io.Reader) ([]DamageEvent, error) {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), 1024*1024)
	ready := false
	events := []DamageEvent{}
	last := map[string]int{}
	for s.Scan() {
		line := strings.TrimSuffix(s.Text(), "\r")
		if line == "g_test_damage ready version=1" {
			ready = true
			continue
		}
		if !strings.HasPrefix(line, "sv_test_damage ") && !strings.HasPrefix(line, "g_test_damage ") {
			continue
		}
		m := damagePattern.FindStringSubmatch(line)
		if m == nil {
			return nil, fmt.Errorf("malformed or unsupported server damage event")
		}
		n := make([]int, 15)
		for _, i := range []int{1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} {
			v, err := strconv.Atoi(m[i])
			if err != nil {
				return nil, fmt.Errorf("damage integer: %w", err)
			}
			n[i] = v
		}
		shot := uint64(0)
		if m[17] != "" {
			var err error
			shot, err = strconv.ParseUint(m[17], 10, 32)
			if err != nil {
				return nil, err
			}
		}
		e := DamageEvent{Map: m[3], Spawncount: n[1], Frame: n[2], GameFrame: n[4], Attacker: n[5], Target: n[6], Inflictor: n[7], Mod: n[8], HealthBefore: n[9], HealthAfter: n[10], Take: n[11], Armor: n[12], Power: n[13], Protection: n[14], TargetClass: m[15], AttackerClass: m[16], Shot: uint32(shot)}
		if e.Frame < 1 || e.HealthBefore-e.Take != e.HealthAfter {
			return nil, fmt.Errorf("inconsistent damage event")
		}
		key := fmt.Sprintf("%s/%d", e.Map, e.Spawncount)
		if e.Frame < last[key] {
			return nil, fmt.Errorf("nonmonotonic server damage")
		}
		last[key] = e.Frame
		events = append(events, e)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if !ready {
		return nil, fmt.Errorf("server damage telemetry not confirmed")
	}
	return events, nil
}

// ServerOutcome describes effects in (observation.frame, next.frame]. It is a
// temporal join, not proof that the last command caused a hit or was executed.
// Delayed projectile damage belongs to its effect window. Nothing is a victory.
type ServerOutcome struct {
	JointDeathEvents     []DamageEvent          `json:"joint_death_events,omitempty"`
	Version              string                 `json:"version"`
	Worker               string                 `json:"worker"`
	Episode              string                 `json:"episode"`
	Step                 int                    `json:"step"`
	Available            bool                   `json:"available"`
	Reason               string                 `json:"reason,omitempty"`
	MonsterHealthDamage  int                    `json:"monster_health_damage"`
	ReceivedHealthDamage int                    `json:"received_health_damage"`
	SelfHealthDamage     int                    `json:"self_health_damage"`
	TeammateHealthDamage int                    `json:"teammate_health_damage"`
	MonsterKills         int                    `json:"monster_kills"`
	Deaths               int                    `json:"deaths"`
	Events               []DamageEvent          `json:"events"`
	Score                *float64               `json:"reward_score"`
	ProjectileMisses     *ProjectileMissOutcome `json:"projectile_misses,omitempty"`
	HitscanMisses        *HitscanMissOutcome    `json:"hitscan_misses,omitempty"`
}

type DamageJoiner struct {
	Events []DamageEvent
	Used   map[int]bool
}

func (j *DamageJoiner) Join(s *Step) ServerOutcome {
	o := ServerOutcome{Version: ServerOutcomeVersion, Worker: s.Worker, Episode: s.Episode, Step: s.Index, Events: []DamageEvent{}}
	if s.Next == nil || s.Next.Identity.Frame != s.Observation.Identity.Frame+1 || !sameWorld(s) {
		o.Reason = "no_consecutive_world_window"
		return o
	}
	o.Available = true
	if j.Used == nil {
		j.Used = map[int]bool{}
	}
	id := s.Observation.Identity
	for i, e := range j.Events {
		if e.Map != id.Map || e.Spawncount != id.Spawncount || e.Frame <= id.Frame || e.Frame > s.Next.Identity.Frame || (e.Attacker != id.Actor && e.Target != id.Actor) {
			continue
		}
		j.Used[i] = true
		o.add(e, id.Actor)
	}
	return o
}

func (o *ServerOutcome) add(e DamageEvent, actor int) {
	o.Events = append(o.Events, e)
	damage := min(max(e.HealthBefore, 0), e.Take)
	killed := e.HealthBefore > 0 && e.HealthAfter <= 0
	if e.Target == actor {
		o.ReceivedHealthDamage += damage
		if killed {
			o.Deaths++
		}
	}
	if e.Attacker == actor {
		switch {
		case e.Target == actor:
			o.SelfHealthDamage += damage
		case strings.HasPrefix(e.TargetClass, "monster_"):
			o.MonsterHealthDamage += damage
			if killed {
				o.MonsterKills++
			}
		case e.TargetClass == "player":
			o.TeammateHealthDamage += damage
		}
	}
}

func (j *DamageJoiner) JoinNative(s *Step, p *NativeStep) ServerOutcome {
	o := ServerOutcome{Version: "server_step_effects_v1", Worker: s.Worker, Episode: s.Episode, Step: s.Index, Events: []DamageEvent{}}
	if s.Next == nil {
		o.Reason = "no_next_observation"
		return o
	}
	o.Available = true
	if j.Used == nil {
		j.Used = map[int]bool{}
	}
	for _, index := range p.DamageIndexes {
		e := j.Events[index]
		if e.Attacker != p.Actor && e.Target != p.Actor {
			continue
		}
		j.Used[index] = true
		o.add(e, p.Actor)
	}
	return o
}

func sameWorld(s *Step) bool {
	a, b := s.Observation.Identity, s.Next.Identity
	return a.Map == b.Map && a.Spawncount == b.Spawncount && a.Connection == b.Connection && a.Actor == b.Actor && a.Life == b.Life
}
