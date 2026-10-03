package bot

import "q2coopbot/internal/quake"

// Shoot only with the nonexplosive, unlimited Blaster. Stop after an observed
// button movement; the parent task still waits for the actual door effect.
func (p *Planner) buttonShotCommand(s quake.Snapshot, prev, cmd quake.UserCmd) (quake.UserCmd, bool) {
	task := p.button
	if task == nil || task.action != "shoot" || task.phase == "approach" {
		return cmd, false
	}
	cmd.Forward, cmd.Side, cmd.Up, cmd.Buttons = 0, 0, 0, 0
	p.World.Command.MoveSource = "button"
	p.World.Command.AimSource = "button"
	p.World.Command.Skill = "button_" + task.phase
	g := p.World.Geometry
	if g == nil {
		return cmd, true
	}
	bounds, ok := g.Model(task.buttonModel)
	if !ok {
		return cmd, true
	}
	var target quake.Vec3
	observed := false
	for _, mover := range s.Movers {
		if mover.Model == task.buttonModel {
			for axis := range target {
				target[axis] = (bounds.Min[axis]+bounds.Max[axis])/2 + mover.Origin[axis]
			}
			observed = true
		}
	}
	if !observed {
		p.World.Command.LimitReason = "button_unobserved"
		return cmd, true
	}
	from := s.EyePoint()
	// UDP userinfo fixes center hand; native Blaster starts eight below the eye.
	from[2] -= 8
	cmd.Yaw = quake.YawTo(from, target, s.DeltaAngles[1])
	cmd.Pitch = quake.PitchTo(from, target, s.DeltaAngles[0])
	p.World.Command.AimPoint = &target
	if task.phase == "wait_effect" {
		p.World.Command.LimitReason = "button_wait_effect"
		return cmd, true
	}
	if s.Weapon != "Blaster" {
		p.World.Command.LimitReason = "button_requires_blaster"
		return cmd, true
	}
	if !g.ClearShot(from, target) || g.DoorShotBlocked(s.Movers, from, target) {
		p.World.Command.LimitReason = "button_shot_blocked"
		return cmd, true
	}
	for _, mover := range s.Movers {
		if mover.Model != task.buttonModel && !g.MoverHullClear(mover, from, target) {
			p.World.Command.LimitReason = "button_shot_blocked"
			return cmd, true
		}
	}
	if s.Teammate != nil && (teammateBlocksShot(from, target, *s.Teammate) || p.teammateEntersProjectile(s, from, target)) {
		p.World.Command.LimitReason = "friendly_line_of_fire"
		return cmd, true
	}
	if s.Frame-task.aimSince < 3 || absAngle(cmd.Yaw-prev.Yaw) > 256 || absAngle(cmd.Pitch-prev.Pitch) > 256 {
		p.World.Command.LimitReason = "button_aim_settling"
		return cmd, true
	}
	if task.shots >= 3 || task.shotAt != 0 && s.Frame-task.shotAt < 10 {
		p.World.Command.LimitReason = "button_wait_effect"
		return cmd, true
	}
	cmd.Buttons = 1
	task.shots++
	task.shotAt = s.Frame
	if p.World.Campaign != nil && p.World.Campaign.Button != nil {
		p.World.Campaign.Button.Shots = task.shots
	}
	return cmd, true
}

func absAngle(value int16) int {
	result := int(value)
	if result < 0 {
		return -result
	}
	return result
}
