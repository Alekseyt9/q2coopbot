package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func (p *Planner) combatNavigationContext() *policy.NavigationContext {
	s := p.World.Snapshot
	context := &policy.NavigationContext{Goal: p.World.Goal, Status: p.World.Navigation}
	local := func(at quake.Vec3) *quake.Vec3 {
		v := quake.Vec3{at[0] - s.Self[0], at[1] - s.Self[1], at[2] - s.Self[2]}
		return &v
	}
	if p.hasGoal {
		context.GoalRelative = local(p.goalPoint)
		if p.routeKnown && p.routeIndex >= 0 && p.routeIndex < len(p.route) {
			context.WaypointRelative = local(p.route[p.routeIndex].Position)
		}
	}
	if s.LastTeammate != nil && s.TeammateAgeFrames != nil && *s.TeammateAgeFrames >= 0 {
		age := *s.TeammateAgeFrames
		context.LastTeammateRelative = local(*s.LastTeammate)
		context.LastTeammateAgeFrames = &age
	}
	return context
}
