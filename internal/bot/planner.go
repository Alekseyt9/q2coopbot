package bot

import (
	"fmt"
	"log"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"q2coopbot/internal/quake"
)

type World struct {
	GrenadePrediction *GrenadePrediction `json:"grenade_prediction,omitempty"`
	ResourceYield     *ResourceYield     `json:"resource_yield,omitempty"`
	Pickup            *PickupAttempt     `json:"pickup,omitempty"`
	SearchRoute       *SearchRouteCheck  `json:"search_route,omitempty"`
	Map               string             `json:"map"`
	Geometry          *quake.MapInfo     `json:"geometry,omitempty"`
	AASLoaded         bool               `json:"aas_loaded"`
	Areas             int                `json:"areas"`
	Reachabilities    int                `json:"reachabilities"`
	Navigation        string             `json:"navigation"`
	GeometryStatus    string             `json:"geometry_status"`
	Goal              string             `json:"goal"`
	SearchTarget      *quake.Vec3        `json:"search_target,omitempty"`
	SearchAttempt     *SearchAttempt     `json:"search_attempt,omitempty"`
	TeammateSound     *TeammateSoundCue  `json:"teammate_sound,omitempty"`
	TeammateMotion    *TeammateMotion    `json:"teammate_motion,omitempty"`
	TeammateEvidence  *TeammateEvidence  `json:"teammate_evidence,omitempty"`
	Strategy          *StrategyDecision  `json:"strategy,omitempty"`
	Tactic            *TacticalDecision  `json:"tactic,omitempty"`
	Route             []quake.Waypoint   `json:"route,omitempty"`
	Jump              *JumpTrace         `json:"jump_plan,omitempty"`
	Elevator          string             `json:"elevator,omitempty"`
	Command           CommandDecision    `json:"command"`
	LaserEvidence     []LaserEvidence    `json:"laser_evidence,omitempty"`
	Snapshot          quake.Snapshot     `json:"snapshot"`
	Updated           time.Time          `json:"updated"`
}
type Planner struct {
	grenadeThrow              *grenadeThrow
	grenadeRequestFrame       int
	grenadeRequestMap         string
	TestDisableHandGrenade    bool
	testDoorPassSpeed         float64
	doorPrevious              quake.Snapshot
	TestDisableProjectileLead bool
	railAim                   railAim
	enemyMotion               map[int]enemyMotion
	shotTeammateMotion        shotTeammateMotion
	urgentRetreat             urgentRetreat
	resources                 map[int]*ResourceMemory
	healthStarted             int
	pickup                    *pickupTask
	pickupBanned              map[quake.Vec3]int
	pickupNext                int
	healthActive              bool
	healthTarget, healthLast  quake.Vec3
	healthAt                  int
	healthBanned              map[quake.Vec3]int
	deathFrame                int
	jump                      *jumpFlight
	TestDisableProbe          bool
	testSetupHold             bool
	TestDisableSearch         bool
	Nav                       *quake.Navigator
	AASDir                    string
	GameClock                 bool
	TestNoAAS                 bool
	TestNoBSP                 bool
	TestPartialBSP            bool
	TestHideDoor53            bool
	button                    *buttonTask
	buttonCooldown            int
	World                     World
	lastSelf                  quake.Vec3
	lastProgress              time.Time
	routeAt                   time.Time
	target                    quake.Vec3
	route                     []quake.Waypoint
	routeIndex                int
	routeKnown                bool
	routeOK                   bool
	lastObserved              quake.Vec3
	observed                  bool
	detourUntil               time.Time
	detourSide                int16
	failures                  int
	goalPoint                 quake.Vec3
	hasGoal                   bool
	decision                  *StrategyDecision
	tactic                    *TacticalDecision
	elevator                  *elevatorRide
	bridgeLink                *bridgeLink
	probeTarget               *quake.Vec3
	searchAttempt             *SearchAttempt
	probeAttempted            bool
	probeProgressFrame        int
	probeLastSelf             quake.Vec3
	lastSeenSelf              quake.Vec3
	lastSeenSelfKnown         bool
	searchApproachStarted     bool
	longSearch                bool
	teammateSoundCue          *TeammateSoundCue
	teammateEvidence          *TeammateEvidence
	respawnRegroup            *respawnRegroup
	deathPoint                *quake.Vec3
	machinegunBurst           machinegunBurst
	laserEvidence             map[quake.Vec3]*laserMemory
	cornerHistory             []quake.Vec3
	cornerEscapeTarget        quake.Vec3
	cornerEscapeStart         quake.Vec3
	cornerEscapeUntil         int
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
	p.laserEvidence = nil
	p.Nav = nil
	p.button = nil
	p.deathFrame = 0
	p.healthActive = false
	p.healthBanned = nil
	p.resources = nil
	p.pickup, p.pickupBanned, p.pickupNext = nil, nil, 0
	p.jump = nil
	p.cornerHistory = nil
	p.cornerEscapeUntil = 0
	p.buttonCooldown = 0
	p.failures = 0
	p.decision = nil
	p.tactic = nil
	p.route = nil
	p.routeIndex = 0
	p.routeKnown = false
	p.elevator = nil
	p.bridgeLink = nil
	p.respawnRegroup = nil
	p.deathPoint = nil
	p.probeTarget = nil
	p.searchAttempt = nil
	p.probeAttempted = false
	p.probeProgressFrame = 0
	p.lastSeenSelfKnown = false
	p.searchApproachStarted = false
	p.longSearch = false
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
	p.World.ResourceYield = nil
	p.observeResources(s)
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
	if p.testSetupHold {
		// Setup may place the actor/bot or move doors. Start release with a
		// route from the observed placement, without an artificial stuck jump.
		p.routeKnown = false
		p.lastSelf = s.Self
		p.lastProgress = now
		p.detourUntil = time.Time{}
		p.failures = 0
	}
	if p.World.Geometry.HasCollision() {
		for i := range s.Enemies {
			from, to := s.EyePoint(), s.Enemies[i].AimPoint()
			clear := p.World.Geometry.ClearShot(from, to) && !p.World.Geometry.DoorShotBlocked(s.Movers, from, to)
			s.Enemies[i].ClearShot = &clear
		}
	}
	p.observeEnemyMotion(s)
	p.observeShotTeammateMotion(s)
	p.observeUrgentRetreat(s)
	previous := p.World.Snapshot
	p.observeLasers(previous, s)
	p.observeRespawnRegroup(previous, s)
	p.doorPrevious = previous
	if previous.Frame > 0 && !previous.OnGround && s.OnGround && p.elevator == nil && !p.routeOK {
		// AAS may have no route from an airborne area. Refresh it as soon
		// as a supported landing gives us a valid starting area again.
		p.routeKnown = false
	}
	if previous.Frame > 0 && previous.Health <= 0 && s.Health > 0 {
		p.deathFrame = 0
		p.bridgeLink = nil
		p.routeKnown = false
		p.jump = nil
		p.elevator = nil
		p.button = nil
		p.lastProgress = time.Time{}
	}
	p.World.Snapshot = s
	s.Pickups = append([]quake.Object(nil), s.Pickups...)
	sort.Slice(s.Pickups, func(i, j int) bool {
		a, b := quake.Distance(s.Self, s.Pickups[i].Origin), quake.Distance(s.Self, s.Pickups[j].Origin)
		if a == b {
			return s.Pickups[i].ID < s.Pickups[j].ID
		}
		return a < b
	})
	// Drop the interaction before any early return from hidden-player search.
	p.validateButtonOwner(s)
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
	regrouping := false
	standalonePickup := false
	goal := quake.Vec3{}
	if s.Teammate == nil {
		searchGoal, ok := p.respawnRegroupGoal(s)
		searchKind := "regroup_after_respawn"
		regrouping = ok
		if !ok {
			p.elevator = nil
			searchGoal, searchKind, ok = p.hiddenTeammateGoal(s)
		}
		p.World.SearchAttempt = p.searchAttempt
		if !ok {
			searchGoal, ok = p.pickupGoal(s)
			if !ok {
				p.routeKnown = false
				return
			}
			standalonePickup = true
			searchKind = "collect_item"
		}
		searching = !standalonePickup
		goal = searchGoal
		p.World.Goal = searchKind
		if searching {
			p.World.SearchTarget = &searchGoal
		}
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
		p.longSearch = false
		p.lastSeenSelf = s.Self
		p.lastSeenSelfKnown = true
		if previousGoal == "search_last_seen" || previousGoal == "probe_last_seen" || previousGoal == "wait_for_teammate" || previousGoal == "regroup_after_respawn" {
			p.routeKnown = false
		}
	}
	if s.Teammate != nil {
		goal = *s.Teammate
	}
	p.hasGoal = true
	if !searching && s.Health < 45 {
		if health, ok := p.healthGoal(s); ok {
			goal = health
			p.World.Goal = "recover_health"
		}
	}
	if searching { /* the last known point is a search target, not a visible teammate */
	} else if p.World.Goal == "recover_health" || standalonePickup { /* keep resource objective */
	} else if quake.Horizontal(s.Self, goal) < followStandOff && math.Abs(s.Self[2]-goal[2]) < 40 && !p.bridgeNeedsApproach(goal) && !p.bridgeLinkNeedsExit() {
		p.World.Goal = "cover_teammate"
	} else {
		p.World.Goal = "follow_teammate"
	}
	if !searching && !standalonePickup && p.World.Strategy != nil {
		switch p.World.Strategy.Choice {
		case "recover":
			if health, ok := p.healthGoal(s); ok {
				goal = health
				p.World.Goal = "recover_health"
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
	if !searching && !standalonePickup && p.World.Tactic != nil && p.World.Tactic.Action == "recover" {
		if health, ok := p.healthGoal(s); ok {
			goal = health
			p.World.Goal = "recover_health"
		}
	}
	goal = p.budgetHealthGoal(s, goal)
	if item, ok := p.pickupGoal(s); ok {
		goal = item
		p.World.Goal = "collect_item"
	}
	if previousGoal != p.World.Goal {
		// A stalled health/search task is not evidence that the new follow
		// route is stuck. Inheriting that timer immediately launches a detour
		// jump, which can throw the bot off the stairs before it tries the route.
		p.lastProgress = now
		p.lastSelf = s.Self
		p.detourUntil = time.Time{}
		p.failures = 0
	}
	p.goalPoint = goal
	// A button is a subtask of following this player, not an override for a
	// newly selected health objective or an already restored close contact.
	if p.World.Goal != "follow_teammate" {
		p.cancelButtonTask(s.Frame)
		if p.bridgeLink != nil {
			p.bridgeLink = nil
			p.routeKnown = false
		}
	}
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
	if teleported {
		p.bridgeLink = nil
		p.routeKnown = false
	}
	p.lastObserved = s.Self
	p.observed = true
	goalChanged := p.elevator == nil && (quake.Horizontal(goal, p.target) > 80 || math.Abs(goal[2]-p.target[2]) > 32)
	if p.elevator != nil && teleported {
		p.elevator = nil
		p.routeKnown = false
	}
	if p.elevator != nil && s.Frame%10 == 0 && p.mayLeaveElevatorForBridge() {
		if r, _, ok := p.deployedBridgeRoute(); ok && routeLength(s.Self, r, goal) < routeLength(s.Self, p.route[p.routeIndex:], goal) {
			p.routeKnown = false
		}
	}
	if !p.routeKnown || goalChanged && p.bridgeLink == nil || p.elevator == nil && p.bridgeLink == nil && now.Sub(p.routeAt) > 4*time.Second || teleported {
		// A ride owns the board/exit pair of this route. A replacement route
		// must not inherit its state or suppress subsequent route refreshes.
		p.elevator = nil
		p.routeAt = now
		p.target = goal
		p.routeIndex = 0
		if searching && !regrouping {
			p.route, p.routeOK = p.Nav.SearchRoute(s.Self, goal)
			if !p.routeOK {
				// Retain diagnostics for a graph path requiring forbidden travel;
				// the search-policy check below still prevents its execution.
				p.route, p.routeOK = p.Nav.Route(s.Self, goal)
			}
		} else {
			p.route, p.routeOK = p.directCrouchRoute(s, goal)
			if !p.routeOK {
				p.route, p.routeOK = p.bridgeRoute()
			}
			if !p.routeOK {
				p.route, p.routeOK = p.Nav.Route(s.Self, goal)
			}
			// A visible player can take priority immediately after spawning,
			// while the spawn area still lacks outgoing AAS links.
			if !p.routeOK && (regrouping || p.World.Goal == "follow_teammate") {
				p.route, p.routeOK = p.regroupEntryRoute(s, goal)
			}
			if !p.routeOK {
				p.route, p.routeOK = p.localFlatRoute(s, goal)
			}
		}
		p.routeKnown = true
		p.bridgeLink = nil
		if r, task, ok := p.deployedBridgeRoute(); ok && (!p.routeOK || routeLength(s.Self, r, goal) < routeLength(s.Self, p.route, goal)) {
			p.route, p.routeOK, p.bridgeLink = r, true, task
		}
	}
	if p.routeOK {
		for p.routeIndex < len(p.route) && p.route[p.routeIndex].Kind != 11 && quake.Horizontal(s.Self, p.route[p.routeIndex].Position) <= 10 {
			// Walking reaches can climb successive 16-unit steps. Being
			// horizontally close does not mean the bot has climbed one.
			// On a descent the bot may already be supported above the AAS
			// sample by a lift, so retain the wider downward tolerance.
			if p.route[p.routeIndex].Kind == 2 && p.route[p.routeIndex].Position[2]-s.Self[2] > 2 ||
				math.Abs(s.Self[2]-p.route[p.routeIndex].Position[2]) > 64 {
				break
			}
			p.routeIndex++
		}
	}
	routeOK := p.routeOK
	if searching && !regrouping {
		maxTravel := 640.0
		if p.World.Goal == "probe_last_seen" {
			maxTravel = 320
		} else if p.longSearch {
			maxTravel = 1400
		}
		check := &SearchRouteCheck{FromArea: p.Nav.AreaFor(s.Self), ToArea: p.Nav.AreaFor(goal), GraphRouteFound: p.routeOK, Limit: maxTravel, Reason: "route_missing"}
		if p.routeOK {
			travel, reason := checkSearchRoute(p.route[p.routeIndex:], s.Self, goal, maxTravel)
			if p.longSearch && reason == "ready" {
				for _, wp := range p.route[p.routeIndex:] {
					if wp.Kind != 2 && wp.Kind != 3 {
						reason = "requires_nonwalking_transition"
						break
					}
				}
			}
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

func (p *Planner) commandAt(prev quake.UserCmd, now time.Time) (result quake.UserCmd) {
	cmd := quake.UserCmd{Yaw: prev.Yaw, Msec: 50}
	p.World.Command = CommandDecision{MoveSource: "none", AimSource: "none"}
	p.World.Jump = nil
	s := p.World.Snapshot
	defer func() { result = p.guardHandGrenade(s, p.limitMachinegunBurst(s, p.limitLaserMovement(s, result))) }()
	if !isRailgun(s.Weapon) || s.Health <= 0 {
		p.railAim = railAim{}
	}
	if p.World.Map == "" || s.Frame == 0 {
		p.World.Command.LimitReason = "no_frame"
		return cmd
	}
	if p.World.Updated.IsZero() || now.Sub(p.World.Updated) > 300*time.Millisecond {
		p.World.Command.LimitReason = "stale_observation"
		return cmd
	}
	if s.Health <= 0 {
		p.jump = nil
		if p.deathFrame == 0 || s.Frame < p.deathFrame {
			p.deathFrame = s.Frame
		}
		p.World.Command.LimitReason = "respawn_wait"
		if age := s.Frame - p.deathFrame; age >= 10 && age%5 == 0 {
			cmd.Buttons = 1
			p.World.Command.LimitReason = "respawn_request"
		}
		return cmd
	}
	if p.World.GeometryStatus == "unavailable" || p.World.GeometryStatus == "incomplete" {
		p.World.Command.LimitReason = "bsp_" + p.World.GeometryStatus
		p.World.Command.MoveLimitReason = p.World.Command.LimitReason
		return cmd
	}
	if flight, active := p.jumpCommand(cmd); active {
		return flight
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
		// Rank only confirmed lines of fire. A nearer enemy behind a wall
		// must not hide a farther visible target from combat arbitration.
		if s.Enemies[i].ClearShot == nil || !*s.Enemies[i].ClearShot {
			continue
		}
		d := quake.Distance(s.Self, s.Enemies[i].Origin)
		if d < best {
			best = d
			enemy = &s.Enemies[i]
		}
	}
	combatRange := 650.0
	if isRailgun(s.Weapon) {
		combatRange = 1000
	}
	if p.World.Goal != "search_last_seen" && p.World.Goal != "probe_last_seen" && p.World.Goal != "touch_button" && p.World.Goal != "approach_button" && tactic != "follow" && tactic != "recover" && enemy != nil && best < combatRange && (s.Ammo > 0 || strings.Contains(strings.ToLower(s.Weapon), "blast")) && enemy.ClearShot != nil && *enemy.ClearShot {
		from, to := s.EyePoint(), enemy.AimPoint()
		to, leadSeconds := p.projectileAim(s, *enemy)
		p.World.Command.AimPoint = &to
		p.World.Command.AimEntity = enemy.ID
		p.World.Command.LeadSeconds = leadSeconds
		safetyEnd := to
		if isRailgun(s.Weapon) {
			safetyEnd = railEnd(from, to)
		}
		if s.Teammate != nil && teammateBlocksShot(from, safetyEnd, *s.Teammate) {
			p.World.Command.LimitReason = "friendly_line_of_fire"
			p.railAim = railAim{}
		} else if p.teammateEntersProjectile(s, from, to) {
			p.World.Command.LimitReason = "friendly_projectile_crossing"
		} else {
			cmd.Yaw = quake.YawTo(from, to, s.DeltaAngles[1])
			cmd.Pitch = quake.PitchTo(from, to, s.DeltaAngles[0])
			p.World.Command.AimSource = "enemy"
			ready := true
			if isRailgun(s.Weapon) {
				cmd, ready = p.railCommand(s, *enemy, prev, cmd)
				if !ready {
					p.World.Command.LimitReason = "rail_aim_settling"
				}
			}
			if ready {
				cmd.Buttons = 1
			}
		}
	}
	if yield, ok := p.playerYieldCommand(s, cmd); ok {
		return yield
	}
	profile := combatSpacing(s)
	p.World.Command.CombatSpacing = profile
	if (p.World.Goal == "cover_teammate" || p.World.Goal == "follow_teammate") && profile != nil && profile.Distance < profile.Minimum+32 && s.Teammate != nil && quake.Distance(s.Self, *s.Teammate) <= combatLeash(profile) && (tactic == "" || tactic == "attack" || tactic == "retreat") {
		if profile.NeedSpace {
			return p.combatRetreat(cmd, profile)
		}
		p.World.Command.MoveLimitReason = "combat_spacing_hold"
		return cmd
	}
	if tactic == "attack" {
		if p.World.Command.LimitReason == "" {
			p.World.Command.LimitReason = "tactic_" + tactic + "_stationary"
		}
		return cmd
	}
	if exit, active := p.platformExitCommand(cmd); active {
		return exit
	}
	if bridge, active := p.bridgeCommand(cmd); active {
		return bridge
	}
	if bridge, active := p.bridgeLinkCommand(cmd); active {
		return bridge
	}
	if p.planWalkOff() || p.planNearbyWalkOff() {
		flight, _ := p.jumpCommand(quake.UserCmd{Yaw: cmd.Yaw})
		return flight
	}
	if p.World.Goal != "follow_teammate" && p.World.Goal != "collect_item" && p.World.Goal != "recover_health" && p.World.Goal != "touch_button" && p.World.Goal != "approach_button" && p.World.Goal != "search_last_seen" && p.World.Goal != "probe_last_seen" && p.World.Goal != "regroup_after_respawn" || p.World.Navigation != "ready" && p.World.Navigation != "direct_clear" || !p.hasGoal {
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
		if quake.Horizontal(s.Self, wp.Position) > 10 || wp.Kind == 2 && wp.Position[2]-s.Self[2] > 2 || math.Abs(s.Self[2]-wp.Position[2]) > 64 {
			target = wp.Position
			jump = wp.Jump
			break
		}
	}
	if escape, ok := p.regroupCornerEscape(s); ok {
		target = escape
		jump = false
		p.World.Command.Skill = "route_corner_escape"
	}
	if quake.Horizontal(s.Self, target) < 10 && target[2]-s.Self[2] <= 2 {
		if p.World.Command.LimitReason == "" {
			p.World.Command.LimitReason = "at_waypoint"
		}
		return cmd
	}
	moveSpeedLimit := 400.0
	// Brake before turns leading into a walk-off reach. The movement guard
	// checks the requested direction, but momentum survives a sharp turn.
	for i, wp := range p.World.Route {
		if wp.Kind == 11 {
			break
		}
		// A gap-jump approach can also contain kind 7. Brake only for
		// a paired downward reach; reducing jump run-up speed causes falls.
		if wp.Kind == 7 && i+1 < len(p.World.Route) && p.World.Route[i+1].Kind == 7 && wp.ToArea == p.World.Route[i+1].ToArea && wp.Position[2]-p.World.Route[i+1].Position[2] >= 24 && quake.Horizontal(s.Self, wp.Position) < 256 && math.Abs(s.Self[2]-wp.Position[2]) < 32 {
			moveSpeedLimit = 120
			break
		}
	}
	dx, dy := target[0]-s.Self[0], target[1]-s.Self[1]
	// Preserve a nearby rise until the native standing hull has climbed it.
	// A short checked step avoids skipping an 8-unit stair before a turn.
	if s.OnGround && len(p.World.Route) > 0 && p.World.Route[0].Kind == 2 &&
		target[2]-s.Self[2] > 2 && target[2]-s.Self[2] <= 18 &&
		quake.Horizontal(s.Self, target) <= 10 &&
		p.World.Geometry.GroundMoveHazardStep(p.Nav, s.Self, target[0]-s.Self[0], target[1]-s.Self[1], 8) == "" {
		moveSpeedLimit = 80
	}
	probeDistance := moveSpeedLimit / 10
	if moveSpeedLimit < 400 {
		probeDistance = math.Min(probeDistance, math.Hypot(dx, dy))
	}
	hazard := p.World.Geometry.GroundMoveHazardStep(p.Nav, s.Self, dx, dy, probeDistance)
	// At the measured base1 second rise, a full-speed tick skips beyond a
	// nearby grounded AAS walking reach. Other ramps must keep their existing
	// jump planning; taking short steps there can spoil a safe run-up.
	if s.OnGround && s.Map == "base1" && (hazard == "no_ground_support" || hazard == "static_hull_blocked") && len(p.World.Route) > 0 &&
		p.World.Route[0].Kind == 2 && (p.World.Route[0].ToArea == 2452 || p.World.Route[0].ToArea == 2453) &&
		target[2]-s.Self[2] > 0 && target[2]-s.Self[2] <= 18 &&
		quake.Horizontal(s.Self, target) <= 24 &&
		p.World.Geometry.GroundMoveHazardStep(p.Nav, s.Self, dx, dy, 8) == "" {
		moveSpeedLimit, probeDistance = 80, 8
		hazard = ""
	}
	if s.OnGround && (hazard == "no_ground_support" || hazard == "static_hull_blocked" && (p.blockedDropApproach() || p.blockedRiseApproach())) {
		verified := p.planWalkOff()
		if !verified && hazard == "no_ground_support" && !p.lastProgress.IsZero() && now.Sub(p.lastProgress) >= 700*time.Millisecond && math.Hypot(s.SelfVelocity[0], s.SelfVelocity[1]) < 80 {
			// A controlled short descent is a recovery option after the
			// normal lift approach has genuinely stopped making progress.
			verified = p.planShortWalkDown()
		}
		if !verified && p.blockedRiseApproach() {
			verified = p.planRampJump()
		} else if !verified {
			verified = p.planGapJump()
		}
		if verified {
			flight, _ := p.jumpCommand(cmd)
			return flight
		}
		if hazard == "no_ground_support" {
			p.World.Command.MoveLimitReason = "no_verified_landing"
			p.World.Command.LimitReason = "no_verified_landing"
			return cmd
		}
	}
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
	if cmd.Up == 0 {
		if ax, ay, ok := p.resourceObstacleStep(s, dx, dy); ok {
			dx, dy = ax, ay
			moveSpeedLimit, probeDistance = 80, 16
			p.World.Command.MoveSource = "resource_obstacle_avoid"
			p.World.Command.MoveLimitReason = "resource_obstacle_avoid"
			p.routeKnown = false
		}
	}
	if s.OnGround && cmd.Up == 0 {
		// A short verified ducked sweep can pass a low static ceiling.
		// Keep the standing-hull door guard below: ducking must not bypass doors.
		if math.Abs(target[2]-s.Self[2]) <= 2 &&
			p.World.Geometry.GroundMoveHazardStep(p.Nav, s.Self, dx, dy, 16) == "static_hull_blocked" &&
			p.World.Geometry.CrouchStepClear(s.Self, dx, dy, 16) {
			cmd.Up = -200
			moveSpeedLimit = math.Min(moveSpeedLimit, 80)
			p.World.Command.Skill = "crouch_passage"
		}
	}
	if s.OnGround && cmd.Up == 0 {
		probeStep := probeDistance
		if p.World.Goal == "touch_button" {
			probeStep = 8
		} else if p.World.Goal == "probe_last_seen" {
			probeStep = 16
		}
		hazard := p.World.Geometry.GroundMoveHazardStep(p.Nav, s.Self, dx, dy, probeStep)
		// A full tick may span two stair risers. Slow down only when a
		// shorter, fully checked ground step is available.
		if hazard == "static_hull_blocked" && probeStep > 16 && p.World.Geometry.GroundMoveHazardStep(p.Nav, s.Self, dx, dy, 16) == "" {
			hazard = ""
			moveSpeedLimit = 160
		}
		if hazard != "" {
			if hazard == "static_hull_blocked" {
				if sx, sy, ok := p.regroupCornerStep(s, target); ok {
					p.routeKnown = false
					p.World.Command.MoveSource = "route_corner_bypass"
					p.World.Command.Skill = "route_corner_bypass"
					p.World.Command.LimitReason = "verified_corner_step"
					return worldMove(cmd, s, sx, sy, 80, false)
				}
			}
			p.World.Command.MoveSource = "none"
			p.World.Command.MoveLimitReason = hazard
			if p.World.Command.LimitReason == "" {
				p.World.Command.LimitReason = hazard
			}
			return cmd
		}
	}
	if s.OnGround {
		hazard := p.World.Geometry.DoorMoveHazard(s.Movers, s.Self, dx, dy)
		// A clear short approach need not clear an entire full-speed tick.
		// Recheck static and dynamic hulls every tick, and request only half
		// the checked distance in 100 ms, even for a distant waypoint.
		if hazard != "" && cmd.Up == 0 && p.World.Command.MoveSource == "route" && p.World.Geometry.GroundMoveHazardStep(p.Nav, s.Self, dx, dy, 16) == "" {
			if _, shortHazard := p.World.Geometry.DoorMoveBlockStep(s.Movers, s.Self, dx, dy, 16); shortHazard == "" {
				hazard = ""
				moveSpeedLimit = math.Min(moveSpeedLimit, 80)
				p.World.Command.MoveLimitReason = "door_short_approach"
			}
		}
		if hazard != "" {
			cmd.Up = 0
			if hazard == "dynamic_door_blocked" {
				if bypass, ok := p.doorRouteBypass(s); ok {
					p.World.Command.MoveSource = "door_route_bypass"
					p.World.Command.Skill = "door_route_bypass"
					p.World.Command.MoveLimitReason = "verified_door_route_bypass"
					return worldMove(cmd, s, bypass[0]-s.Self[0], bypass[1]-s.Self[1], 80, p.World.Command.AimSource == "enemy")
				}
			}
			p.World.Command.MoveSource = "none"
			p.World.Command.MoveLimitReason = hazard
			if p.World.Command.LimitReason == "" {
				p.World.Command.LimitReason = hazard
			}
			return cmd
		}
	}
	speed := 400.0
	if p.testDoorPassSpeed > 0 {
		speed = math.Min(speed, p.testDoorPassSpeed)
	}
	if !jump && cmd.Up == 0 {
		speed = math.Min(speed, math.Hypot(dx, dy)*10)
	}
	if p.World.Goal == "touch_button" {
		speed = 80
	} else if p.World.Goal == "probe_last_seen" {
		speed = 160
	}
	cmd = worldMove(cmd, s, dx, dy, math.Min(speed, moveSpeedLimit), p.World.Command.AimSource == "enemy")
	if p.World.Command.AimSource != "enemy" {
		p.World.Command.AimSource = "route"
	}
	return cmd
}
