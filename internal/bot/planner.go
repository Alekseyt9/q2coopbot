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
	SearchRoute      *SearchRouteCheck `json:"search_route,omitempty"`
	Map              string            `json:"map"`
	Geometry         *quake.MapInfo    `json:"geometry,omitempty"`
	AASLoaded        bool              `json:"aas_loaded"`
	Areas            int               `json:"areas"`
	Reachabilities   int               `json:"reachabilities"`
	Navigation       string            `json:"navigation"`
	GeometryStatus   string            `json:"geometry_status"`
	Goal             string            `json:"goal"`
	SearchTarget     *quake.Vec3       `json:"search_target,omitempty"`
	SearchAttempt    *SearchAttempt    `json:"search_attempt,omitempty"`
	TeammateSound    *TeammateSoundCue `json:"teammate_sound,omitempty"`
	TeammateMotion   *TeammateMotion   `json:"teammate_motion,omitempty"`
	TeammateEvidence *TeammateEvidence `json:"teammate_evidence,omitempty"`
	Strategy         *StrategyDecision `json:"strategy,omitempty"`
	Tactic           *TacticalDecision `json:"tactic,omitempty"`
	Route            []quake.Waypoint  `json:"route,omitempty"`
	Elevator         string            `json:"elevator,omitempty"`
	Command          CommandDecision   `json:"command"`
	Snapshot         quake.Snapshot    `json:"snapshot"`
	Updated          time.Time         `json:"updated"`
}
type Planner struct {
	TestDisableProbe      bool
	testSetupHold         bool
	TestDisableSearch     bool
	Nav                   *quake.Navigator
	AASDir                string
	GameClock             bool
	TestNoAAS             bool
	TestNoBSP             bool
	TestPartialBSP        bool
	TestHideDoor53        bool
	button                *buttonTask
	buttonCooldown        int
	World                 World
	lastSelf              quake.Vec3
	lastProgress          time.Time
	routeAt               time.Time
	target                quake.Vec3
	route                 []quake.Waypoint
	routeIndex            int
	routeKnown            bool
	routeOK               bool
	lastObserved          quake.Vec3
	observed              bool
	detourUntil           time.Time
	detourSide            int16
	failures              int
	goalPoint             quake.Vec3
	hasGoal               bool
	decision              *StrategyDecision
	tactic                *TacticalDecision
	elevator              *elevatorRide
	probeTarget           *quake.Vec3
	searchAttempt         *SearchAttempt
	probeAttempted        bool
	probeProgressFrame    int
	probeLastSelf         quake.Vec3
	lastSeenSelf          quake.Vec3
	lastSeenSelfKnown     bool
	searchApproachStarted bool
	teammateSoundCue      *TeammateSoundCue
	teammateEvidence      *TeammateEvidence
}

// setTestGroundEdgeGoal bypasses route selection only for the live edge fixture.
// The normal command path still decides whether the proposed step is safe.
func (p *Planner) setTestGroundEdgeGoal() {
	s := p.World.Snapshot
	if p.World.Map != "base1" || !s.OnGround || math.Abs(s.Self[0]+88) > 8 || math.Abs(s.Self[1]-40) > 8 {
		return
	}
	p.World.Goal = "follow_teammate"
	p.World.Navigation = "direct_clear"
	p.World.Route = nil
	p.goalPoint = quake.Vec3{s.Self[0], s.Self[1] - 200, s.Self[2]}
	p.hasGoal = true
	p.detourUntil = time.Time{}
}

// setTestDoorGoal exercises the normal command guard with a direct door approach.
func (p *Planner) setTestDoorGoal() {
	s := p.World.Snapshot
	if p.World.Map != "base2" || !s.OnGround || math.Abs(s.Self[0]-96) > 8 || math.Abs(s.Self[1]+300) > 8 {
		return
	}
	p.World.Goal = "follow_teammate"
	p.World.Navigation = "direct_clear"
	p.World.Route = nil
	p.goalPoint = quake.Vec3{96, -160, s.Self[2]}
	p.hasGoal = true
	p.detourUntil = time.Time{}
}

func (p *Planner) setTestDoorPassGoal() {
	s := p.World.Snapshot
	if p.World.Map != "base2" || !s.OnGround {
		return
	}
	p.World.Goal = "follow_teammate"
	p.World.Navigation = "direct_clear"
	p.World.Route = nil
	p.goalPoint = quake.Vec3{112, -800, s.Self[2]}
	p.hasGoal = true
	p.detourUntil = time.Time{}
}

// setTestButtonGoal exercises a touch-operated button through ordinary movement.
func (p *Planner) setTestButtonGoal() {
	s := p.World.Snapshot
	if p.World.Map != "base2" || !s.OnGround {
		return
	}
	button, action, ok := p.World.Geometry.ButtonForDoor(33)
	if !ok || action != "touch" {
		return
	}
	model, ok := p.World.Geometry.Model(button.Model)
	if !ok {
		return
	}
	p.World.Goal = "touch_button"
	p.World.Navigation = "direct_clear"
	p.World.Route = nil
	p.goalPoint = quake.Vec3{(model.Min[0] + model.Max[0]) / 2, model.Max[1] + 28, s.Self[2]}
	p.hasGoal = true
	p.detourUntil = time.Time{}
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
	p.World = World{Map: name, Navigation: "aas_missing", GeometryStatus: "unavailable"}
	p.Nav = nil
	p.button = nil
	p.buttonCooldown = 0
	p.failures = 0
	p.decision = nil
	p.tactic = nil
	p.route = nil
	p.routeIndex = 0
	p.routeKnown = false
	p.elevator = nil
	p.probeTarget = nil
	p.searchAttempt = nil
	p.probeAttempted = false
	p.probeProgressFrame = 0
	p.lastSeenSelfKnown = false
	p.searchApproachStarted = false
	p.teammateSoundCue = nil
	p.teammateEvidence = nil
	p.observed = false
	p.lastProgress = time.Time{}
	p.detourUntil = time.Time{}
	p.detourSide = 200
	if name == "" {
		return
	}
	if !p.TestNoBSP {
		if info, e := quake.LoadMap(root, name); e == nil {
			if p.TestPartialBSP {
				info.Models = nil
			}
			p.World.Geometry = &info
			if info.MovementComplete() {
				p.World.GeometryStatus = "ready"
			} else {
				p.World.GeometryStatus = "incomplete"
			}
			log.Printf("map=%s BSP status=%s entities=%d brushes=%d", name, p.World.GeometryStatus, len(info.Entities), info.Brushes)
		} else {
			log.Printf("map=%s BSP unavailable: %v", name, e)
		}
	} else {
		log.Printf("map=%s BSP disabled by test fixture", name)
	}
	if p.TestNoAAS {
		log.Printf("map=%s AAS disabled by test fixture", name)
		return
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
	if p.TestHideDoor53 && s.Map == "base2" {
		visible := make([]quake.Mover, 0, len(s.Movers))
		for _, mover := range s.Movers {
			if mover.Model != 53 {
				visible = append(visible, mover)
			}
		}
		s.Movers = visible
	}
	now := p.navigationNow(s.Frame)
	if p.World.Geometry.HasCollision() {
		for i := range s.Enemies {
			from, to := s.Self, s.Enemies[i].Origin
			from[2] += 22
			to[2] += 22
			clear := p.World.Geometry.ClearShot(from, to) && !p.World.Geometry.DoorShotBlocked(s.Movers, from, to)
			s.Enemies[i].ClearShot = &clear
		}
	}
	previous := p.World.Snapshot
	p.World.Snapshot = s
	p.updateTeammateSoundCue(s)
	p.updateTeammateMotion(s)
	p.updateTeammateEvidence(previous, s)
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
	previousGoal := p.World.Goal
	p.World.Goal = "wait_for_teammate"
	p.World.SearchTarget = nil
	p.World.SearchAttempt = nil
	p.World.SearchRoute = nil
	p.World.Route = nil
	p.World.Elevator = ""
	p.hasGoal = false
	searching := false
	if s.Teammate == nil {
		p.elevator = nil
		searchGoal, searchKind, ok := p.hiddenTeammateGoal(s)
		p.World.SearchAttempt = p.searchAttempt
		if !ok {
			p.routeKnown = false
			return
		}
		searching = true
		p.World.Goal = searchKind
		p.World.SearchTarget = &searchGoal
		if previousGoal != searchKind {
			p.routeKnown = false
		}
	} else {
		if p.searchAttempt != nil && p.searchAttempt.State == "completed" && p.searchAttempt.EndFrame < s.Frame {
			p.searchAttempt = nil
		}
		p.finishSearchAttempt(s.Frame, "reacquired")
		p.World.SearchAttempt = p.searchAttempt
		p.probeTarget = nil
		p.probeAttempted = false
		p.searchApproachStarted = false
		p.lastSeenSelf = s.Self
		p.lastSeenSelfKnown = true
		if previousGoal == "search_last_seen" || previousGoal == "probe_last_seen" || previousGoal == "wait_for_teammate" {
			p.routeKnown = false
		}
	}
	goal := quake.Vec3{}
	if searching {
		goal = *p.World.SearchTarget
	} else {
		goal = *s.Teammate
	}
	p.hasGoal = true
	for _, pickup := range s.Pickups {
		if searching {
			break
		}
		if s.Health < 45 && strings.Contains(pickup.Class, "health") && quake.Horizontal(s.Self, pickup.Origin) < 300 {
			goal = pickup.Origin
			p.World.Goal = "recover_health"
			break
		}
	}
	if searching { /* the last known point is a search target, not a visible teammate */
	} else if p.World.Goal == "recover_health" { /* keep health objective */
	} else if quake.Horizontal(s.Self, goal) < 100 && math.Abs(s.Self[2]-goal[2]) < 40 {
		p.World.Goal = "cover_teammate"
	} else {
		p.World.Goal = "follow_teammate"
	}
	if !searching && p.World.Strategy != nil {
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
	if !searching && p.World.Tactic != nil && p.World.Tactic.Action == "recover" {
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
		if p.World.Geometry.HasCollision() && quake.Horizontal(s.Self, goal) < 256 && math.Abs(s.Self[2]-goal[2]) < 40 {
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
	goalChanged := p.elevator == nil && (quake.Horizontal(goal, p.target) > 80 || math.Abs(goal[2]-p.target[2]) > 32)
	if p.elevator != nil && teleported {
		p.elevator = nil
		p.routeKnown = false
	}
	if !p.routeKnown || goalChanged || p.elevator == nil && now.Sub(p.routeAt) > 4*time.Second || teleported {
		p.routeAt = now
		p.target = goal
		p.routeIndex = 0
		if searching {
			p.route, p.routeOK = p.Nav.SearchRoute(s.Self, goal)
			if !p.routeOK {
				// Retain diagnostics for a graph path requiring forbidden travel;
				// the search-policy check below still prevents its execution.
				p.route, p.routeOK = p.Nav.Route(s.Self, goal)
			}
		} else {
			p.route, p.routeOK = p.Nav.Route(s.Self, goal)
		}
		p.routeKnown = true
	}
	if p.routeOK {
		for p.routeIndex < len(p.route) && p.route[p.routeIndex].Kind != 11 && quake.Horizontal(s.Self, p.route[p.routeIndex].Position) <= 10 && math.Abs(s.Self[2]-p.route[p.routeIndex].Position[2]) <= 64 {
			p.routeIndex++
		}
	}
	routeOK := p.routeOK
	if searching {
		maxTravel := 640.0
		if p.World.Goal == "probe_last_seen" {
			maxTravel = 320
		}
		check := &SearchRouteCheck{FromArea: p.Nav.AreaFor(s.Self), ToArea: p.Nav.AreaFor(goal), GraphRouteFound: p.routeOK, Limit: maxTravel, Reason: "route_missing"}
		if p.routeOK {
			travel, reason := checkSearchRoute(p.route[p.routeIndex:], s.Self, goal, maxTravel)
			check.Travel, check.Reason = &travel, reason
			routeOK = reason == "ready"
		}
		p.World.SearchRoute = check
	}
	if searching && p.World.Goal == "probe_last_seen" && !routeOK {
		p.finishSearchAttempt(s.Frame, "route_unavailable")
		p.probeTarget = nil
		p.World.SearchAttempt = p.searchAttempt
		p.World.SearchTarget = nil
		p.World.Goal = "wait_for_teammate"
		p.World.Navigation = "unreachable"
		p.hasGoal = false
		return
	}
	if routeOK {
		p.World.Route = p.route[p.routeIndex:]
		p.World.Navigation = "ready"
	} else {
		p.World.Navigation = "unreachable"
	}
	if p.elevator != nil {
		p.World.Elevator = p.elevator.stage
	}
	if quake.Horizontal(s.Self, p.lastSelf) > 12 {
		p.lastProgress = now
		p.lastSelf = s.Self
		p.failures = 0
	} else if p.lastProgress.IsZero() {
		p.lastProgress = now
		p.lastSelf = s.Self
	}
	if p.elevator == nil && p.button == nil && p.World.Goal == "follow_teammate" && now.Sub(p.lastProgress) > 2500*time.Millisecond && now.After(p.detourUntil) {
		p.failures++
		p.detourSide = -p.detourSide
		p.detourUntil = now.Add(800 * time.Millisecond)
		p.lastProgress = now
		p.routeKnown = false
		log.Printf("navigation stuck map=%s self=%v failures=%d", s.Map, s.Self, p.failures)
	}
	if !searching && p.World.Navigation == "unreachable" && p.World.Geometry.HasCollision() && quake.Horizontal(s.Self, goal) < 256 && math.Abs(s.Self[2]-goal[2]) < 40 {
		from, to := s.Self, goal
		from[2] += 18
		to[2] += 18
		if p.World.Geometry.ClearShot(from, to) {
			p.World.Navigation = "direct_clear"
		}
	}
}
func (p *Planner) command(prev quake.UserCmd) quake.UserCmd {
	return p.commandAt(prev, time.Now())
}

func (p *Planner) commandAt(prev quake.UserCmd, now time.Time) quake.UserCmd {
	cmd := quake.UserCmd{Yaw: prev.Yaw, Msec: 50}
	p.World.Command = CommandDecision{MoveSource: "none", AimSource: "none"}
	s := p.World.Snapshot
	if p.World.Map == "" || s.Frame == 0 {
		p.World.Command.LimitReason = "no_frame"
		return cmd
	}
	if p.World.Updated.IsZero() || now.Sub(p.World.Updated) > 300*time.Millisecond {
		p.World.Command.LimitReason = "stale_observation"
		return cmd
	}
	if s.Health <= 0 {
		p.World.Command.LimitReason = "dead"
		return cmd
	}
	if p.World.GeometryStatus == "unavailable" || p.World.GeometryStatus == "incomplete" {
		p.World.Command.LimitReason = "bsp_" + p.World.GeometryStatus
		p.World.Command.MoveLimitReason = p.World.Command.LimitReason
		return cmd
	}
	if p.elevator != nil && p.routeIndex < len(p.route) && p.route[p.routeIndex].ElevatorPhase == "board" {
		cmd = p.elevatorCommand(cmd, p.route[p.routeIndex])
		p.World.Command = CommandDecision{MoveSource: "elevator", AimSource: "elevator", Skill: "elevator", LimitReason: p.World.Elevator}
		return cmd
	}
	tactic := ""
	if p.World.Tactic != nil {
		tactic = p.World.Tactic.Action
	}
	if tactic == "hold" {
		p.World.Command.LimitReason = "tactic_hold"
		return cmd
	}
	p.applyButtonTask(s)
	var enemy *quake.Object
	best := math.Inf(1)
	for i := range s.Enemies {
		d := quake.Distance(s.Self, s.Enemies[i].Origin)
		if d < best {
			best = d
			enemy = &s.Enemies[i]
		}
	}
	if p.World.Goal != "search_last_seen" && p.World.Goal != "probe_last_seen" && p.World.Goal != "touch_button" && p.World.Goal != "approach_button" && tactic != "follow" && tactic != "recover" && enemy != nil && best < 650 && (s.Ammo > 0 || strings.Contains(strings.ToLower(s.Weapon), "blast")) && enemy.ClearShot != nil && *enemy.ClearShot {
		from, to := s.Self, enemy.Origin
		from[2] += 22
		to[2] += 22
		if s.Teammate != nil && teammateBlocksShot(from, to, *s.Teammate) {
			p.World.Command.LimitReason = "friendly_line_of_fire"
		} else {
			cmd.Yaw = quake.YawTo(from, to, s.DeltaAngles[1])
			cmd.Pitch = quake.PitchTo(from, to, s.DeltaAngles[0])
			cmd.Buttons = 1
			p.World.Command.AimSource = "enemy"
		}
	}
	if tactic == "attack" {
		if p.World.Command.LimitReason == "" {
			p.World.Command.LimitReason = "tactic_attack_stationary"
		}
		return cmd
	}
	if p.World.Goal != "follow_teammate" && p.World.Goal != "recover_health" && p.World.Goal != "touch_button" && p.World.Goal != "approach_button" && p.World.Goal != "search_last_seen" && p.World.Goal != "probe_last_seen" || p.World.Navigation != "ready" && p.World.Navigation != "direct_clear" || !p.hasGoal {
		if p.World.Command.LimitReason == "" {
			p.World.Command.LimitReason = "no_movement_goal"
		}
		return cmd
	}
	if p.routeIndex < len(p.route) && p.route[p.routeIndex].ElevatorPhase == "board" {
		cmd = p.elevatorCommand(cmd, p.route[p.routeIndex])
		p.World.Command = CommandDecision{MoveSource: "elevator", AimSource: "elevator", Skill: "elevator", LimitReason: p.World.Elevator}
		return cmd
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
		if p.World.Command.LimitReason == "" {
			p.World.Command.LimitReason = "at_waypoint"
		}
		return cmd
	}
	dx, dy := target[0]-s.Self[0], target[1]-s.Self[1]
	p.World.Command.MoveSource = "route"
	if jump || target[2]-s.Self[2] > 32 {
		cmd.Up = 200
	}
	if p.navigationNow(s.Frame).Before(p.detourUntil) {
		cmd.Up = 200
		distance := math.Hypot(dx, dy)
		if distance > 0 {
			ux, uy := dx/distance, dy/distance
			dx = ux + uy*float64(p.detourSide)/400
			dy = uy - ux*float64(p.detourSide)/400
			p.World.Command.MoveSource = "detour"
		}
	}
	if s.OnGround && cmd.Up == 0 {
		probeStep := 40.0
		if p.World.Goal == "touch_button" {
			probeStep = 8
		} else if p.World.Goal == "probe_last_seen" {
			probeStep = 16
		}
		if hazard := p.World.Geometry.GroundMoveHazardStep(p.Nav, s.Self, dx, dy, probeStep); hazard != "" {
			p.World.Command.MoveSource = "none"
			p.World.Command.MoveLimitReason = hazard
			if p.World.Command.LimitReason == "" {
				p.World.Command.LimitReason = hazard
			}
			return cmd
		}
	}
	if s.OnGround {
		if hazard := p.World.Geometry.DoorMoveHazard(s.Movers, s.Self, dx, dy); hazard != "" {
			cmd.Up = 0
			p.World.Command.MoveSource = "none"
			p.World.Command.MoveLimitReason = hazard
			if p.World.Command.LimitReason == "" {
				p.World.Command.LimitReason = hazard
			}
			return cmd
		}
	}
	speed := 400.0
	if p.World.Goal == "touch_button" {
		speed = 80
	} else if p.World.Goal == "probe_last_seen" {
		speed = 160
	}
	cmd = worldMove(cmd, s, dx, dy, speed, cmd.Buttons != 0)
	if cmd.Buttons == 0 {
		p.World.Command.AimSource = "route"
	}
	return cmd
}
