package bot

import (
	"fmt"
	"log"
	"math"
	"path/filepath"
	"strings"
	"time"

	"q2coopbot/internal/quake"
)

type World struct {
	Map            string            `json:"map"`
	Geometry       *quake.MapInfo    `json:"geometry,omitempty"`
	AASLoaded      bool              `json:"aas_loaded"`
	Areas          int               `json:"areas"`
	Reachabilities int               `json:"reachabilities"`
	Navigation     string            `json:"navigation"`
	Goal           string            `json:"goal"`
	Strategy       *StrategyDecision `json:"strategy,omitempty"`
	Tactic         *TacticalDecision `json:"tactic,omitempty"`
	Route          []quake.Waypoint  `json:"route,omitempty"`
	Elevator       string            `json:"elevator,omitempty"`
	Snapshot       quake.Snapshot    `json:"snapshot"`
	Updated        time.Time         `json:"updated"`
}
type Planner struct {
	Nav          *quake.Navigator
	AASDir       string
	GameClock    bool
	World        World
	lastSelf     quake.Vec3
	lastProgress time.Time
	routeAt      time.Time
	target       quake.Vec3
	route        []quake.Waypoint
	routeIndex   int
	routeKnown   bool
	routeOK      bool
	lastObserved quake.Vec3
	observed     bool
	detourUntil  time.Time
	detourSide   int16
	failures     int
	goalPoint    quake.Vec3
	hasGoal      bool
	decision     *StrategyDecision
	tactic       *TacticalDecision
	elevator     *elevatorRide
}

func (p *Planner) navigationNow(frame int) time.Time {
	if p.GameClock {
		return time.Unix(0, int64(frame)*int64(100*time.Millisecond))
	}
	return time.Now()
}

func (p *Planner) applyDecision(d StrategyDecision) {
	p.decision = &d
	log.Printf("system2 choice=%s latency=%dms reason=%q", d.Choice, d.LatencyMS, d.Reason)
}
func (p *Planner) applyTactic(d TacticalDecision) {
	p.tactic = &d
	log.Printf("system1 action=%s latency=%dms", d.Action, d.LatencyMS)
}

func (p *Planner) setMap(name, root string) {
	if p.World.Map == name {
		return
	}
	p.World = World{Map: name, Navigation: "aas_missing"}
	p.Nav = nil
	p.failures = 0
	p.decision = nil
	p.tactic = nil
	p.route = nil
	p.routeIndex = 0
	p.routeKnown = false
	p.elevator = nil
	p.observed = false
	p.lastProgress = time.Time{}
	p.detourUntil = time.Time{}
	p.detourSide = 200
	if name == "" {
		return
	}
	if info, e := quake.LoadMap(root, name); e == nil {
		p.World.Geometry = &info
		log.Printf("map=%s BSP entities=%d brushes=%d", name, len(info.Entities), info.Brushes)
	} else {
		log.Printf("map=%s BSP unavailable: %v", name, e)
	}
	paths := []string{filepath.Join(p.AASDir, name+".aas"), filepath.Join(root, "maps", name+".aas")}
	var n *quake.Navigator
	var e error
	for _, path := range paths {
		candidate, loadErr := quake.LoadAAS(path)
		if loadErr != nil {
			e = loadErr
			continue
		}
		edges := 0
		for _, list := range candidate.Edges {
			edges += len(list)
		}
		if edges == 0 {
			e = fmt.Errorf("AAS contains no traversable reaches: %s", path)
			continue
		}
		n = candidate
		break
	}
	if n == nil {
		log.Printf("map=%s AAS unavailable: %v", name, e)
		return
	}
	p.Nav = n
	p.World.AASLoaded = true
	p.World.Areas = len(n.Areas)
	for _, edges := range n.Edges {
		p.World.Reachabilities += len(edges)
	}
	p.World.Navigation = "ready"
	log.Printf("map=%s AAS areas=%d reachabilities=%d", name, p.World.Areas, p.World.Reachabilities)
}
func (p *Planner) update(s quake.Snapshot, root string) {
	p.setMap(s.Map, root)
	now := p.navigationNow(s.Frame)
	if p.World.Geometry.HasCollision() {
		for i := range s.Enemies {
			from, to := s.Self, s.Enemies[i].Origin
			from[2] += 22
			to[2] += 22
			clear := p.World.Geometry.ClearShot(from, to)
			s.Enemies[i].ClearShot = &clear
		}
	}
	p.World.Snapshot = s
	p.World.Updated = time.Now()
	if p.decision != nil && time.Since(p.decision.At) < 8*time.Second {
		p.World.Strategy = p.decision
	} else {
		p.World.Strategy = nil
	}
	if p.tactic != nil && time.Since(p.tactic.At) < 3*time.Second {
		p.World.Tactic = p.tactic
	} else {
		p.World.Tactic = nil
	}
	p.World.Goal = "wait_for_teammate"
	p.World.Route = nil
	p.hasGoal = false
	if s.Teammate == nil {
		return
	}
	goal := *s.Teammate
	p.hasGoal = true
	for _, pickup := range s.Pickups {
		if s.Health < 45 && strings.Contains(pickup.Class, "health") && quake.Horizontal(s.Self, pickup.Origin) < 300 {
			goal = pickup.Origin
			p.World.Goal = "recover_health"
			break
		}
	}
	if p.World.Goal == "recover_health" { /* keep health objective */
	} else if quake.Horizontal(s.Self, goal) < 100 && math.Abs(s.Self[2]-goal[2]) < 64 {
		p.World.Goal = "cover_teammate"
	} else {
		p.World.Goal = "follow_teammate"
	}
	if p.World.Strategy != nil {
		switch p.World.Strategy.Choice {
		case "recover":
			for _, pickup := range s.Pickups {
				if strings.Contains(pickup.Class, "health") && quake.Horizontal(s.Self, pickup.Origin) < 300 {
					goal = pickup.Origin
					p.World.Goal = "recover_health"
					break
				}
			}
		case "engage":
			for _, enemy := range s.Enemies {
				if enemy.ClearShot != nil && *enemy.ClearShot {
					p.World.Goal = "engage_enemy"
					break
				}
			}
		case "reposition":
			if p.World.Goal != "recover_health" {
				p.World.Goal = "follow_teammate"
			}
		}
	}
	if p.World.Tactic != nil && p.World.Tactic.Action == "recover" {
		for _, pickup := range s.Pickups {
			if strings.Contains(pickup.Class, "health") && quake.Horizontal(s.Self, pickup.Origin) < 300 {
				goal = pickup.Origin
				p.World.Goal = "recover_health"
				break
			}
		}
	}
	p.goalPoint = goal
	if p.Nav == nil {
		if p.World.Geometry.HasCollision() && quake.Horizontal(s.Self, goal) < 256 && math.Abs(s.Self[2]-goal[2]) < 64 {
			from, to := s.Self, goal
			from[2] += 18
			to[2] += 18
			if p.World.Geometry.ClearShot(from, to) {
				p.World.Navigation = "direct_clear"
			}
		}
		return
	}
	teleported := p.observed && quake.Horizontal(s.Self, p.lastObserved) > 256
	p.lastObserved = s.Self
	p.observed = true
	goalChanged := quake.Horizontal(goal, p.target) > 80 || math.Abs(goal[2]-p.target[2]) > 80
	if p.elevator != nil && (goalChanged || teleported) {
		p.elevator = nil
		p.routeKnown = false
	}
	if !p.routeKnown || goalChanged || p.elevator == nil && now.Sub(p.routeAt) > 4*time.Second || teleported {
		p.routeAt = now
		p.target = goal
		p.routeIndex = 0
		p.route, p.routeOK = p.Nav.Route(s.Self, goal)
		p.routeKnown = true
	}
	if p.routeOK {
		for p.routeIndex < len(p.route) && p.route[p.routeIndex].Kind != 11 && quake.Horizontal(s.Self, p.route[p.routeIndex].Position) <= 10 && math.Abs(s.Self[2]-p.route[p.routeIndex].Position[2]) <= 64 {
			p.routeIndex++
		}
		p.World.Route = p.route[p.routeIndex:]
		p.World.Navigation = "ready"
	} else {
		p.World.Navigation = "unreachable"
	}
	if quake.Horizontal(s.Self, p.lastSelf) > 12 {
		p.lastProgress = now
		p.lastSelf = s.Self
		p.failures = 0
	} else if p.lastProgress.IsZero() {
		p.lastProgress = now
		p.lastSelf = s.Self
	}
	if p.elevator == nil && p.World.Goal == "follow_teammate" && now.Sub(p.lastProgress) > 2500*time.Millisecond && now.After(p.detourUntil) {
		p.failures++
		p.detourSide = -p.detourSide
		p.detourUntil = now.Add(800 * time.Millisecond)
		p.lastProgress = now
		p.routeKnown = false
		log.Printf("navigation stuck map=%s self=%v failures=%d", s.Map, s.Self, p.failures)
	}
	if p.World.Navigation == "unreachable" && p.World.Geometry.HasCollision() && quake.Horizontal(s.Self, goal) < 256 && math.Abs(s.Self[2]-goal[2]) < 64 {
		from, to := s.Self, goal
		from[2] += 18
		to[2] += 18
		if p.World.Geometry.ClearShot(from, to) {
			p.World.Navigation = "direct_clear"
		}
	}
}
func (p *Planner) command(prev quake.UserCmd) quake.UserCmd {
	cmd := quake.UserCmd{Yaw: prev.Yaw, Msec: 50}
	s := p.World.Snapshot
	if p.World.Map == "" || s.Frame == 0 {
		return cmd
	}
	if s.Health <= 0 {
		return cmd
	}
	tactic := ""
	if p.World.Tactic != nil {
		tactic = p.World.Tactic.Action
	}
	if tactic == "hold" {
		return cmd
	}
	var enemy *quake.Object
	best := math.Inf(1)
	for i := range s.Enemies {
		d := quake.Distance(s.Self, s.Enemies[i].Origin)
		if d < best {
			best = d
			enemy = &s.Enemies[i]
		}
	}
	if tactic != "follow" && tactic != "recover" && enemy != nil && best < 650 && (s.Ammo > 0 || strings.Contains(strings.ToLower(s.Weapon), "blast")) && enemy.ClearShot != nil && *enemy.ClearShot {
		from, to := s.Self, enemy.Origin
		from[2] += 22
		to[2] += 22
		cmd.Yaw = quake.YawTo(from, to, s.DeltaAngles[1])
		cmd.Pitch = quake.PitchTo(from, to, s.DeltaAngles[0])
		cmd.Buttons = 1
	}
	if tactic == "attack" {
		return cmd
	}
	if p.World.Goal != "follow_teammate" && p.World.Goal != "recover_health" || p.World.Navigation != "ready" && p.World.Navigation != "direct_clear" || !p.hasGoal {
		return cmd
	}
	if p.routeIndex < len(p.route) && p.route[p.routeIndex].ElevatorPhase == "board" {
		return p.elevatorCommand(cmd, p.route[p.routeIndex])
	}
	target := p.goalPoint
	jump := false
	for _, wp := range p.World.Route {
		if quake.Horizontal(s.Self, wp.Position) > 10 || math.Abs(s.Self[2]-wp.Position[2]) > 64 {
			target = wp.Position
			jump = wp.Jump
			break
		}
	}
	if quake.Horizontal(s.Self, target) < 10 {
		return cmd
	}
	if cmd.Buttons == 0 {
		cmd.Yaw = quake.YawTo(s.Self, target, s.DeltaAngles[1])
	}
	cmd.Forward = 400
	if jump || target[2]-s.Self[2] > 32 {
		cmd.Up = 200
	}
	if p.navigationNow(s.Frame).Before(p.detourUntil) {
		cmd.Up = 200
		cmd.Side = p.detourSide
	}
	return cmd
}
