package policy

import (
	"math"
	"q2coopbot/internal/quake"
	"sort"
)

type GeometryProbe struct {
	YawOffset         float64    `json:"yaw_offset_degrees"`
	RayDistance       float64    `json:"eye_ray_distance"`
	RayNormal         quake.Vec3 `json:"eye_ray_normal"`
	StandingClearance *float64   `json:"standing_hull_clearance"`
	GroundDrop        *float64   `json:"ground_drop_at_40"`
}
type LocalGeometry struct {
	Frame      int             `json:"frame"`
	RayRange   float64         `json:"ray_range"`
	HullRange  float64         `json:"hull_range"`
	DropRange  float64         `json:"drop_range"`
	Probes     []GeometryProbe `json:"probes"`
	UpDistance float64         `json:"up_eye_distance"`
	GroundDrop *float64        `json:"ground_drop"`
}
type NearbyObject struct {
	ID           int        `json:"id"`
	Class        string     `json:"class"`
	Relative     quake.Vec3 `json:"relative"`
	Distance     float64    `json:"distance"`
	Solid        uint16     `json:"solid"`
	HealthAmount *int       `json:"health_amount"`
}
type NearbyMover struct {
	ID             int        `json:"id"`
	Model          int        `json:"model"`
	RelativeOrigin quake.Vec3 `json:"relative_origin"`
	Angles         quake.Vec3 `json:"angles"`
	ModelMin       quake.Vec3 `json:"model_min"`
	ModelMax       quake.Vec3 `json:"model_max"`
}

type NearbyBeam struct {
	ID            int         `json:"id"`
	Frame         int         `json:"frame"`
	StartRelative quake.Vec3  `json:"start_relative"`
	EndRelative   quake.Vec3  `json:"end_relative"`
	Direction     *quake.Vec3 `json:"direction"`
}

// Static map probes reveal known BSP geometry. Live object positions require
// current PVS plus a clear BSP/observed-door ray. No server state is read here.
func EnrichEnvironment(o *Observation, s quake.Snapshot, g *quake.MapInfo) {
	if !g.HasCollision() {
		return
	}
	local := &LocalGeometry{Frame: s.Frame, RayRange: 256, HullRange: 64, DropRange: 32, Probes: []GeometryProbe{}}
	yaw := degrees(s.ViewAngles[1]) * math.Pi / 180
	eye := s.EyePoint()
	standingValid := g.PlayerMoveClear(s.Self, s.Self)
	for n := 0; n < 8; n++ {
		a := yaw + float64(n)*math.Pi/4
		dx, dy := math.Cos(a), math.Sin(a)
		target := eye
		target[0] += 256 * dx
		target[1] += 256 * dy
		tr := g.TraceProjectile(eye, target)
		if !tr.Valid || tr.StartSolid {
			return
		}
		p := GeometryProbe{YawOffset: float64(n) * 45, RayDistance: tr.Fraction * 256, RayNormal: tr.Normal}
		if standingValid {
			lo, hi := 0.0, 64.0
			end := s.Self
			end[0] += 64 * dx
			end[1] += 64 * dy
			if g.PlayerMoveClear(s.Self, end) {
				lo = 64
			} else {
				for i := 0; i < 7; i++ {
					mid := (lo + hi) / 2
					end = s.Self
					end[0] += mid * dx
					end[1] += mid * dy
					if g.PlayerMoveClear(s.Self, end) {
						lo = mid
					} else {
						hi = mid
					}
				}
			}
			p.StandingClearance = &lo
		}
		at := s.Self
		at[0] += 40 * dx
		at[1] += 40 * dy
		if drop, ok := g.GroundDrop(at, 32); ok {
			p.GroundDrop = &drop
		}
		local.Probes = append(local.Probes, p)
	}
	up := eye
	up[2] += 64
	tr := g.TraceProjectile(eye, up)
	if !tr.Valid || tr.StartSolid {
		return
	}
	local.UpDistance = tr.Fraction * 64
	if drop, ok := g.GroundDrop(s.Self, 32); ok {
		local.GroundDrop = &drop
	}
	o.Geometry = local
	visible := func(at quake.Vec3) bool {
		tr := g.TraceProjectile(eye, at)
		return tr.Valid && !tr.StartSolid && tr.Fraction >= .99999 && !g.DoorShotBlocked(s.Movers, eye, at)
	}
	projectiles := []Enemy{}
	for _, p := range s.Projectiles {
		if quake.Distance(s.Self, p.Origin) > 1024 || !visible(p.Origin) {
			continue
		}
		clear := true
		projectiles = append(projectiles, Enemy{ID: p.ID, Class: p.Class, Relative: relative(p.Origin, s.Self), Distance: quake.Distance(p.Origin, s.Self), ClearShot: &clear})
	}
	sort.Slice(projectiles, func(i, j int) bool {
		if projectiles[i].Distance == projectiles[j].Distance {
			return projectiles[i].ID < projectiles[j].ID
		}
		return projectiles[i].Distance < projectiles[j].Distance
	})
	if len(projectiles) > 16 {
		projectiles = projectiles[:16]
	}
	o.Projectiles = &projectiles
	objects := func(raw []quake.Object) *[]NearbyObject {
		result := []NearbyObject{}
		for _, p := range raw {
			at := p.Origin
			at[2] += 8
			if quake.Distance(s.Self, p.Origin) > 384 || !visible(at) {
				continue
			}
			item := NearbyObject{ID: p.ID, Class: p.Class, Relative: relative(p.Origin, s.Self), Distance: quake.Distance(p.Origin, s.Self), Solid: p.Solid}
			if p.HealthAmount > 0 {
				v := p.HealthAmount
				item.HealthAmount = &v
			}
			result = append(result, item)
		}
		sort.Slice(result, func(i, j int) bool {
			if result[i].Distance == result[j].Distance {
				return result[i].ID < result[j].ID
			}
			return result[i].Distance < result[j].Distance
		})
		if len(result) > 12 {
			result = result[:12]
		}
		return &result
	}
	o.Pickups = objects(s.Pickups)
	o.Props = objects(s.Barrels)
	movers := []NearbyMover{}
	for _, m := range s.Movers {
		model, ok := g.Model(m.Model)
		if !ok {
			continue
		}
		center := quake.Vec3{}
		for axis := range center {
			center[axis] = (model.Min[axis]+model.Max[axis])/2 + m.Origin[axis]
		}
		// Rotating mover bounds need a transformed collision envelope; mask these.
		if m.Angles != (quake.Vec3{}) || quake.Distance(center, s.Self) > 512 || !visible(center) {
			continue
		}
		movers = append(movers, NearbyMover{ID: m.ID, Model: m.Model, RelativeOrigin: relative(m.Origin, s.Self), Angles: m.Angles, ModelMin: model.Min, ModelMax: model.Max})
	}
	sort.Slice(movers, func(i, j int) bool { return movers[i].ID < movers[j].ID })
	if len(movers) > 16 {
		movers = movers[:16]
	}
	o.Movers = &movers
	beams := []NearbyBeam{}
	for _, beam := range s.Beams {
		if beam.Frame != s.Frame {
			continue
		}
		mid := quake.Vec3{}
		for i := range mid {
			mid[i] = (beam.Origin[i] + beam.End[i]) / 2
		}
		if quake.Distance(mid, s.Self) > 512 || !visible(mid) {
			continue
		}
		v := relative(beam.End, beam.Origin)
		length := quake.Distance(beam.End, beam.Origin)
		b := NearbyBeam{ID: beam.ID, Frame: beam.Frame, StartRelative: relative(beam.Origin, s.Self), EndRelative: relative(beam.End, s.Self)}
		if length > 1e-6 {
			direction := quake.Vec3{v[0] / length, v[1] / length, v[2] / length}
			b.Direction = &direction
		}
		beams = append(beams, b)
	}
	sort.Slice(beams, func(i, j int) bool { return beams[i].ID < beams[j].ID })
	if len(beams) > 16 {
		beams = beams[:16]
	}
	o.Beams = &beams
}
