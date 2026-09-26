package bot

import (
	"math"

	"q2coopbot/internal/quake"
)

type buttonTask struct {
	doorModel      int
	buttonModel    int
	initial        quake.Vec3
	stand          quake.Vec3
	touch          quake.Vec3
	phase          string
	started        int
	teammateEntity int
}

func (p *Planner) cancelButtonTask(frame int) {
	if p.button != nil {
		p.button = nil
		p.buttonCooldown = frame + 30
	}
}

func (p *Planner) validateButtonOwner(s quake.Snapshot) {
	if p.button != nil && (s.Health <= 0 || s.Teammate == nil || s.TeammateEntity != p.button.teammateEntity) {
		p.cancelButtonTask(s.Frame)
	}
}

func (p *Planner) selectButtonTask(s quake.Snapshot) *buttonTask {
	if p.World.GeometryStatus != "ready" || p.World.Goal != "follow_teammate" || !s.OnGround ||
		s.Frame < p.buttonCooldown || quake.Horizontal(s.Self, p.goalPoint) > 256 {
		return nil
	}
	if p.World.Navigation != "ready" && p.World.Navigation != "unreachable" {
		return nil
	}
	if p.World.Navigation == "ready" {
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
	if reason != "dynamic_door_blocked" {
		return nil
	}
	button, action, ok := p.World.Geometry.ButtonForDoor(model)
	if !ok || action != "touch" {
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
	for _, mover := range s.Movers {
		if mover.Model == button.Model {
			buttonObserved = true
			break
		}
	}
	if !buttonObserved {
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
	best := math.Inf(1)
	var chosen [2]quake.Vec3
	for _, candidate := range candidates {
		stand := candidate[0]
		distance := quake.Horizontal(s.Self, stand)
		if distance > 256 || distance >= best || !p.World.Geometry.PlayerMoveClear(s.Self, stand) {
			continue
		}
		if _, ok := p.World.Geometry.GroundDrop(stand, 24); !ok && !p.Nav.GroundedNear(stand) {
			continue
		}
		best, chosen = distance, candidate
	}
	if math.IsInf(best, 1) {
		return nil
	}
	return &buttonTask{doorModel: model, buttonModel: button.Model, initial: doorOrigin,
		stand: chosen[0], touch: chosen[1], phase: "approach", started: s.Frame, teammateEntity: s.TeammateEntity}
}

func (p *Planner) applyButtonTask(s quake.Snapshot) {
	if p.button == nil {
		p.button = p.selectButtonTask(s)
	}
	if p.button == nil {
		return
	}
	task := p.button
	if s.Frame-task.started > 80 || p.World.GeometryStatus != "ready" || s.Teammate == nil {
		p.button = nil
		p.buttonCooldown = s.Frame + 30
		return
	}
	buttonObserved := false
	for _, mover := range s.Movers {
		if mover.Model == task.buttonModel {
			buttonObserved = true
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
	}
	p.World.Goal = "approach_button"
	p.goalPoint = task.stand
	if task.phase == "touch" {
		p.World.Goal = "touch_button"
		p.goalPoint = task.touch
	}
	p.World.Navigation = "direct_clear"
	p.World.Route = nil
	p.hasGoal = true
	p.World.Command.Skill = "button_" + task.phase
}
