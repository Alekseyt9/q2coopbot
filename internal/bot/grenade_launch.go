package bot

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
)

type GrenadeLaunchCheck struct {
	ArmFrame                int     `json:"arm_frame"`
	LaunchFrame             int     `json:"launch_frame"`
	Timer                   float64 `json:"timer"`
	ExpectedSpeed           int     `json:"expected_speed"`
	SpeedError              float64 `json:"speed_error"`
	UpImpulse, RightImpulse float64
	ExplosionFrame          int        `json:"explosion_frame"`
	ExpectedExplosionFrame  int        `json:"expected_explosion_frame"`
	ExplosionPosition       quake.Vec3 `json:"explosion_position"`
}

// Isolated fixture only. Consecutive gunframes reveal activation of grenade_time
// on the tick AFTER entering frame11, and throw on the tick AFTER entering12.
// Do not apply this inference to a grenade first observed mid-preparation.
func checkGrenadeLaunch(rows []quake.Snapshot, c GrenadeCalibration, contact ...bool) (GrenadeLaunchCheck, error) {
	r := GrenadeLaunchCheck{}
	first11 := -1
	var launch *quake.Snapshot
	var last *quake.Object
	for i := range rows {
		s := &rows[i]
		if first11 < 0 && s.GunFrame == 11 && isHandGrenade(s.Weapon) {
			if i == 0 || rows[i-1].Frame+1 != s.Frame || rows[i-1].GunFrame != 10 {
				return r, fmt.Errorf("arming start unknown")
			}
			first11 = s.Frame
			r.ArmFrame = first11 + 1
		}
		for j := range s.Projectiles {
			o := &s.Projectiles[j]
			if o.ID != c.Entity {
				continue
			}
			last = o
			if launch == nil {
				if i == 0 || rows[i-1].Frame+1 != s.Frame || rows[i-1].GunFrame != 12 || s.GunFrame != 13 {
					return r, fmt.Errorf("native launch timing unknown")
				}
				launch = s
				r.LaunchFrame = s.Frame
			}
		}
	}
	if first11 < 0 || launch == nil || last == nil {
		return r, fmt.Errorf("native launch missing")
	}
	r.ExpectedExplosionFrame = r.ArmFrame + 32
	r.Timer = float64(r.ExpectedExplosionFrame-r.LaunchFrame) * 0.1
	if r.Timer <= 0 || r.Timer > 3.2 {
		return r, fmt.Errorf("invalid release timer")
	}
	r.ExpectedSpeed = int(400 + (3-r.Timer)*(400.0/3))
	pitch, yaw := float64(launch.ViewAngles[0])*2*math.Pi/65536, float64(launch.ViewAngles[1])*2*math.Pi/65536
	sp, cp := math.Sincos(pitch)
	sy, cy := math.Sincos(yaw)
	fwd, right, up := quake.Vec3{cp * cy, cp * sy, -sp}, quake.Vec3{sy, -cy, 0}, quake.Vec3{sp * cy, sp * sy, cp}
	v := c.Velocity
	v[2] += float64(launch.Gravity) * 0.1 * float64(c.InitialFrame-r.LaunchFrame+1)
	forward := 0.0
	for i := range v {
		forward += v[i] * fwd[i]
		r.UpImpulse += v[i] * up[i]
		r.RightImpulse += v[i] * right[i]
	}
	r.SpeedError = math.Abs(forward - float64(r.ExpectedSpeed))
	if r.SpeedError > 0.6 || r.UpImpulse < 189.4 || r.UpImpulse > 210.6 || math.Abs(r.RightImpulse) > 10.6 {
		return r, fmt.Errorf("native launch impulse disagrees with timer/jitter")
	}
	viewheight := 22.0
	if launch.Ducked {
		viewheight = -2
	}
	for i := range fwd {
		start := launch.Self[i] + 8*fwd[i]
		if i == 2 {
			start += viewheight - 8
		}
		nominal := start + 0.1*(float64(r.ExpectedSpeed)*fwd[i]+200*up[i])
		if i == 2 {
			nominal -= float64(launch.Gravity) * 0.01
		}
		radius := math.Abs(up[i]) + math.Abs(right[i]) + 0.125
		if math.Abs(launch.Projectiles[0].Origin[i]-nominal) > radius {
			return r, fmt.Errorf("center-hand native muzzle offset disagrees")
		}
	}
	count := 0
	for _, s := range rows {
		for _, e := range s.Explosions {
			if e.Kind != 7 && e.Kind != 8 && e.Kind != 17 && e.Kind != 18 {
				continue
			}
			count++
			r.ExplosionFrame = s.Frame
			r.ExplosionPosition = e.Position
		}
	}
	if len(contact) > 0 && contact[0] {
		if count != 1 || r.ExplosionFrame >= r.ExpectedExplosionFrame-5 {
			return r, fmt.Errorf("early native explosion not confirmed")
		}
		return r, nil
	}
	if count != 1 || math.Abs(float64(r.ExplosionFrame-r.ExpectedExplosionFrame)) > 1 || quake.Distance(r.ExplosionPosition, last.Origin) > 16 {
		return r, fmt.Errorf("native fuse explosion not confirmed")
	}
	return r, nil
}
