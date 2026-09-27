package bot

import (
	"math"
	"q2coopbot/internal/quake"
	"strings"
)

type shotTeammateMotion struct {
	velocity       quake.Vec3
	frame, entity  int
	known          bool
	velocityChange float64
}

func (p *Planner) observeShotTeammateMotion(s quake.Snapshot) {
	old := p.World.Snapshot
	previous := p.shotTeammateMotion
	p.shotTeammateMotion = shotTeammateMotion{}
	if s.Teammate == nil || old.Teammate == nil || s.TeammateEntity <= 0 || s.TeammateEntity != old.TeammateEntity || old.Frame+1 != s.Frame || old.Map != s.Map || s.Health <= 0 || old.Health <= 0 {
		return
	}
	v := quake.Vec3{(s.Teammate[0] - old.Teammate[0]) * 10, (s.Teammate[1] - old.Teammate[1]) * 10, 0}
	if math.Hypot(v[0], v[1]) > 400 || math.Abs(s.Teammate[2]-old.Teammate[2]) > 2 {
		return
	}
	p.shotTeammateMotion = shotTeammateMotion{velocity: v, frame: s.Frame, entity: s.TeammateEntity, known: true}
	if previous.known && previous.frame == old.Frame && previous.entity == s.TeammateEntity {
		p.shotTeammateMotion.velocityChange = math.Hypot(v[0]-previous.velocity[0], v[1]-previous.velocity[1])
	}
}

// Constant observed velocity is a reason to defer a shot, never proof of safety.
// Only the baseq2 Blaster/Hyperblaster speed is modeled here. Abrupt future turns,
// other players and projectiles already in flight remain outside this guard.
func (p *Planner) teammateEntersProjectile(s quake.Snapshot, from, to quake.Vec3) bool {
	m := p.shotTeammateMotion
	weapon := strings.ToLower(s.Weapon)
	if (weapon != "blaster" && !strings.Contains(weapon, "/v_hyperb/")) || s.Teammate == nil || !m.known || m.frame != s.Frame || m.entity != s.TeammateEntity {
		return false
	}
	d := quake.Distance(from, to)
	if d < 1 {
		return false
	}
	flight := min(d/1000, 0.5)
	// Sweep the bolt relative to the moving player's center. Using relative
	// time avoids blocking a crossing that happens after the bolt has passed.
	delta := quake.Vec3{}
	for i := 0; i < 3; i++ {
		delta[i] = (to[i]-from[i])*1000/d*flight - m.velocity[i]*flight
	}
	if relativeShotNearTeammate(from, delta, *s.Teammate, 24) {
		return true
	}
	// Braking or turning invalidates the departing-player prediction. Temporarily
	// widen the current body envelope; release it when observed motion settles.
	if m.velocityChange > 8 {
		for i := range delta {
			delta[i] = (to[i] - from[i]) * 1000 / d * flight
		}
		return relativeShotNearTeammate(from, delta, *s.Teammate, 24+min(64, m.velocityChange*flight))
	}
	return false
}

func relativeShotNearTeammate(from, delta, teammate quake.Vec3, radius float64) bool {
	length2 := delta[0]*delta[0] + delta[1]*delta[1]
	if length2 < 1 {
		return false
	}
	u := ((teammate[0]-from[0])*delta[0] + (teammate[1]-from[1])*delta[1]) / length2
	u = max(0, min(1, u))
	x, y, z := from[0]+u*delta[0], from[1]+u*delta[1], from[2]+u*delta[2]
	return math.Hypot(x-teammate[0], y-teammate[1]) <= radius && z >= teammate[2]-28 && z <= teammate[2]+36
}
