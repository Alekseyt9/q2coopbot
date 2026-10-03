package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

type buttonTask struct {
	action         string
	aimSince       int
	shotAt         int
	shots          int
	buttonInitial  quake.Vec3
	chain          []quake.MapEntity
	campaign       bool
	doorModel      int
	buttonModel    int
	initial        quake.Vec3
	stand          quake.Vec3
	touch          quake.Vec3
	phase          string
	started        int
	teammateEntity int
}

type CampaignButtonDecision struct {
	Action      string            `json:"action"`
	Shots       int               `json:"shots,omitempty"`
	DoorModel   int               `json:"door_model"`
	ButtonModel int               `json:"button_model"`
	Phase       string            `json:"phase"`
	Chain       []quake.MapEntity `json:"chain"`
}

func (p *Planner) cancelButtonTask(frame int) {
	if p.button != nil {
		p.button = nil
		p.buttonCooldown = frame + 30
	}
}

func (p *Planner) validateButtonOwner(s quake.Snapshot) {
	if p.button != nil && (s.Health <= 0 || (p.button.campaign && (!p.Campaign || s.Teammate != nil)) || (!p.button.campaign && (s.Teammate == nil || s.TeammateEntity != p.button.teammateEntity))) {
		p.cancelButtonTask(s.Frame)
	}
}

func (p *Planner) selectButtonTask(s quake.Snapshot) *buttonTask {
	campaign := p.World.Goal == "reach_level_exit" && p.Campaign && s.Teammate == nil
	if p.World.GeometryStatus != "ready" || (p.World.Goal != "follow_teammate" && !campaign) || !s.OnGround ||
		s.Frame < p.buttonCooldown || quake.Horizontal(s.Self, p.goalPoint) > 256 {
		return nil
	}
	if p.World.Navigation != "ready" && p.World.Navigation != "unreachable" && p.World.Navigation != "direct_clear" {
		return nil
	}
	if p.World.Navigation == "ready" && !(campaign && s.Self[2]-p.goalPoint[2] > 64) {
		travel, at := 0.0, s.Self
		for _, waypoint := range p.World.Route {
			travel += quake.Horizontal(at, waypoint.Position)
			at = waypoint.Position
		}
		travel += quake.Horizontal(at, p.goalPoint)
		if travel < 1.5*quake.Horizontal(s.Self, p.goalPoint) {
			return nil
		}
	}
	model, reason := p.World.Geometry.DoorMoveBlock(s.Movers, s.Self,
		p.goalPoint[0]-s.Self[0], p.goalPoint[1]-s.Self[1])
	if campaign && reason != "dynamic_door_blocked" {
		for _, mover := range s.Movers {
			if _, action, ok := p.World.Geometry.ButtonForDoor(mover.Model); ok && action == "touch" && !p.World.Geometry.MoverHullClear(mover, s.Self, p.goalPoint) {
				model, reason = mover.Model, "dynamic_door_blocked"
				break
			}
		}
	}
	if reason != "dynamic_door_blocked" {
		return nil
	}
	button, action, ok := p.World.Geometry.ButtonForDoor(model)
	if !ok || (action != "touch" && action != "shoot") {
		return nil
	}
	if p.buttonEffects[buttonEffectKey{button.Model, model}] != nil {
		return nil
	}
	bounds, ok := p.World.Geometry.Model(button.Model)
	if !ok || s.Self[2]+32 < bounds.Min[2] || s.Self[2]-24 > bounds.Max[2] {
		return nil
	}
	var doorOrigin quake.Vec3
	observed := false
	for _, mover := range s.Movers {
		if mover.Model == model {
			doorOrigin, observed = mover.Origin, true
			break
		}
	}
	if !observed {
		return nil
	}
	buttonObserved := false
	var buttonOrigin quake.Vec3
	for _, mover := range s.Movers {
		if mover.Model == button.Model {
			buttonObserved = true
			buttonOrigin = mover.Origin
			break
		}
	}
	if !buttonObserved {
		return nil
	}
	if button.Wait < 0 && quake.Distance(buttonOrigin, button.Origin) > 1 {
		p.rememberButtonEffect(&buttonTask{action: action, buttonModel: button.Model, doorModel: model, buttonInitial: button.Origin, initial: doorOrigin}, s.Frame)
		return nil
	}
	midX, midY := (bounds.Min[0]+bounds.Max[0])/2, (bounds.Min[1]+bounds.Max[1])/2
	z := s.Self[2]
	candidates := [][2]quake.Vec3{
		{{bounds.Min[0] - 36, midY, z}, {bounds.Max[0] + 28, midY, z}},
		{{bounds.Max[0] + 36, midY, z}, {bounds.Min[0] - 28, midY, z}},
		{{midX, bounds.Min[1] - 36, z}, {midX, bounds.Max[1] + 28, z}},
		{{midX, bounds.Max[1] + 36, z}, {midX, bounds.Min[1] - 28, z}},
	}
	if campaign {
		candidates[0][1] = quake.Vec3{bounds.Min[0] - 15.875, midY, z}
		candidates[1][1] = quake.Vec3{bounds.Max[0] + 15.875, midY, z}
		candidates[2][1] = quake.Vec3{midX, bounds.Min[1] - 15.875, z}
		candidates[3][1] = quake.Vec3{midX, bounds.Max[1] + 15.875, z}
	}
	best := math.Inf(1)
	var chosen [2]quake.Vec3
	for _, candidate := range candidates {
		stand := candidate[0]
		distance := quake.Horizontal(s.Self, stand)
		if distance > 256 || distance >= best || !p.World.Geometry.PlayerMoveClear(s.Self, stand) {
			continue
		}
		if _, ok := p.World.Geometry.GroundDrop(stand, 24); !ok && !p.Nav.GroundedNear(stand) {
			live := quake.Mover{Model: model, Origin: doorOrigin}
			if !campaign || !p.stationaryBridge(model, doorOrigin) || !p.bridgeLinkCorridor(s.Self, stand, live) {
				continue
			}
		}
		best, chosen = distance, candidate
	}
	if math.IsInf(best, 1) {
		return nil
	}
	return &buttonTask{action: action, buttonInitial: buttonOrigin, chain: p.World.Geometry.ButtonDoorChain(button.Model, model), campaign: campaign, doorModel: model, buttonModel: button.Model, initial: doorOrigin,
		stand: chosen[0], touch: chosen[1], phase: "approach", started: s.Frame, teammateEntity: s.TeammateEntity}
}

// Native collision with the selected touch button is the intended action.
// Only its collider is excluded, only up to the selected contact point; all
// world/other mover hulls and stationary hatch support remain checked.
func (p *Planner) campaignButtonCorridor(from, to quake.Vec3, live quake.Mover) bool {
	if p.button == nil || !p.button.campaign {
		return false
	}
	q := *p
	if p.button.phase == "touch" {
		if quake.Horizontal(from, to) > quake.Horizontal(from, p.button.touch)+0.01 {
			return false
		}
		q.World.Snapshot.Movers = nil
		for _, mover := range p.World.Snapshot.Movers {
			if mover.Model != p.button.buttonModel {
				q.World.Snapshot.Movers = append(q.World.Snapshot.Movers, mover)
			}
		}
	}
	return q.bridgeLinkCorridor(from, to, live)
}

func (p *Planner) campaignButtonStep(s quake.Snapshot, dx, dy, step float64) bool {
	if p.button == nil || !p.button.campaign || step <= 0 {
		return false
	}
	length := math.Hypot(dx, dy)
	if length == 0 {
		return false
	}
	to := s.Self
	to[0] += dx / length * step
	to[1] += dy / length * step
	for _, live := range s.Movers {
		if live.Model == p.button.doorModel && p.stationaryBridge(live.Model, live.Origin) && p.campaignButtonCorridor(s.Self, to, live) {
			return true
		}
	}
	return false
}

func (p *Planner) applyButtonTask(s quake.Snapshot) {
	p.updateButtonEffects(s)
	if p.button == nil {
		p.button = p.selectButtonTask(s)
	}
	if p.button == nil {
		return
	}
	task := p.button
	if s.Frame-task.started > 80 || p.World.GeometryStatus != "ready" || (!task.campaign && s.Teammate == nil) {
		p.button = nil
		p.buttonCooldown = s.Frame + 30
		return
	}
	buttonObserved := false
	for _, mover := range s.Movers {
		if mover.Model == task.buttonModel {
			buttonObserved = true
			if quake.Distance(mover.Origin, task.buttonInitial) > 1 {
				p.rememberButtonEffect(task, s.Frame)
			}
			if task.action == "shoot" && quake.Distance(mover.Origin, task.buttonInitial) > 1 {
				task.phase = "wait_effect"
			}
			break
		}
	}
	if !buttonObserved {
		p.button = nil
		p.buttonCooldown = s.Frame + 30
		return
	}
	doorObserved := false
	for _, mover := range s.Movers {
		if mover.Model == task.doorModel && quake.Distance(mover.Origin, task.initial) > 60 {
			p.button = nil
			p.buttonCooldown = s.Frame + 30
			return
		}
		if mover.Model == task.doorModel {
			doorObserved = true
		}
	}
	if !doorObserved {
		p.button = nil
		p.buttonCooldown = s.Frame + 30
		return
	}
	if task.phase == "approach" && quake.Horizontal(s.Self, task.stand) < 8 {
		task.phase = "touch"
		if task.action == "shoot" {
			task.phase = "shoot"
			task.aimSince = s.Frame
		}
	}
	p.World.Goal = "approach_button"
	p.goalPoint = task.stand
	if task.phase == "touch" {
		p.World.Goal = "touch_button"
		p.goalPoint = task.touch
	}
	if task.action == "shoot" && task.phase != "approach" {
		p.World.Goal = "shoot_button"
	}
	p.World.Navigation = "direct_clear"
	p.World.Route = nil
	p.hasGoal = true
	p.World.Command.Skill = "button_" + task.phase
	if task.campaign && p.World.Campaign != nil {
		p.World.Campaign.Button = &CampaignButtonDecision{Action: task.action, Shots: task.shots, DoorModel: task.doorModel, ButtonModel: task.buttonModel, Phase: task.phase, Chain: task.chain}
	}
}
