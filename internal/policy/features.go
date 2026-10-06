package policy

import (
	"fmt"
	"math"
	"q2coopbot/internal/quake"
	"sort"
	"strings"
)

const FeatureVersion = "combat_features_v1"
const AimFeatureVersion = "combat_features_v2"
const BBoxFeatureVersion = "combat_features_v3"
const TypedFeatureVersion = "combat_features_v4"
const RecoilFeatureVersion = "combat_features_v5"

// ObservedAimDirection uses only observed protocol bbox and current eye.
// Unknown/non-bbox solids are unavailable, without a guessed body height.
func ObservedAimDirection(o Observation, e Enemy) (quake.Vec3, bool, error) {
	if e.ClearShot == nil || !*e.ClearShot || e.Solid == nil {
		return quake.Vec3{}, false, nil
	}
	s := *e.Solid
	bottom := -float64((s>>5)&31) * 8
	top := float64((s>>10)&63)*8 - 32
	if s == 0 || s == 31 || s&31 == 0 || top <= bottom {
		return quake.Vec3{}, false, nil
	}
	p := (quake.Object{Origin: e.Relative, Solid: s}).AimPoint()
	if o.Ducked {
		p[2] += 2
	} else {
		p[2] -= 22
	}
	n := math.Hypot(math.Hypot(p[0], p[1]), p[2])
	if math.IsNaN(n) || math.IsInf(n, 0) {
		return quake.Vec3{}, false, fmt.Errorf("nonfinite observed aim point")
	}
	if n < 1e-9 {
		return quake.Vec3{}, false, nil
	}
	for i := range p {
		p[i] /= n
	}
	return p, true, nil
}

// FeaturesForVersion preserves v1 exactly. V2 appends eight distance-sorted
// slots: valid mask, sin/cos yaw error, sin/cos pitch error to the observed
// enemy origin from the current eye. This is geometry, not an action or teacher.
func FeaturesForVersion(o Observation, version string) ([]float64, error) {
	if version != FeatureVersion && version != AimFeatureVersion && version != BBoxFeatureVersion && version != TypedFeatureVersion && version != RecoilFeatureVersion {
		return nil, fmt.Errorf("unsupported features %q", version)
	}
	v, err := Features(o)
	if err != nil || version == FeatureVersion {
		return v, err
	}
	enemies := append([]Enemy(nil), o.Enemies...)
	sort.SliceStable(enemies, func(i, j int) bool { return enemies[i].Distance < enemies[j].Distance })
	eye := 22.0
	if o.Ducked {
		eye = -2
	}
	for i := 0; i < 8; i++ {
		if i >= len(enemies) || enemies[i].ClearShot == nil || !*enemies[i].ClearShot {
			v = append(v, 0, 0, 0, 0, 0)
			continue
		}
		r := enemies[i].Relative
		z := r[2] - eye
		length := math.Hypot(math.Hypot(r[0], r[1]), z)
		if math.IsNaN(length) || math.IsInf(length, 0) {
			return nil, fmt.Errorf("nonfinite aim feature")
		}
		if length < 1e-9 {
			v = append(v, 0, 0, 0, 0, 0)
			continue
		}
		yaw := math.Atan2(r[1], r[0]) - degrees(o.ViewAngles[1])*math.Pi/180
		pitch := -math.Atan2(z, math.Hypot(r[0], r[1])) - degrees(o.ViewAngles[0])*math.Pi/180
		v = append(v, 1, math.Sin(yaw), math.Cos(yaw), math.Sin(pitch), math.Cos(pitch))
	}
	if version == BBoxFeatureVersion || version == TypedFeatureVersion || version == RecoilFeatureVersion {
		for i := 0; i < 8; i++ {
			if i >= len(enemies) {
				v = append(v, 0, 0, 0, 0, 0)
				continue
			}
			r, valid, err := ObservedAimDirection(o, enemies[i])
			if err != nil {
				return nil, err
			}
			if !valid {
				v = append(v, 0, 0, 0, 0, 0)
				continue
			}
			yaw := math.Atan2(r[1], r[0]) - degrees(o.ViewAngles[1])*math.Pi/180
			pitch := -math.Atan2(r[2], math.Hypot(r[0], r[1])) - degrees(o.ViewAngles[0])*math.Pi/180
			v = append(v, 1, math.Sin(yaw), math.Cos(yaw), math.Sin(pitch), math.Cos(pitch))
		}
	}
	if version == TypedFeatureVersion || version == RecoilFeatureVersion {
		extra, err := typedFeatures(o, enemies)
		if err != nil {
			return nil, err
		}
		v = append(v, extra...)
	}
	if version == RecoilFeatureVersion {
		if o.KickAngles == nil {
			v = append(v, 0, 0, 0, 0)
		} else {
			v = append(v, 1)
			for _, angle := range *o.KickAngles {
				if math.IsNaN(angle) || math.IsInf(angle, 0) {
					return nil, fmt.Errorf("nonfinite observed kick angle")
				}
				v = append(v, angle/32)
			}
		}
	}
	return v, nil
}

// Features uses only client observations. No seed, frame/actor ID, server
// reward or future state enters the vector. Nil masks remain explicit.
func Features(o Observation) ([]float64, error) {
	v := []float64{}
	put := func(x ...float64) { v = append(v, x...) }
	b := func(x bool) float64 {
		if x {
			return 1
		}
		return 0
	}
	yaw := degrees(o.ViewAngles[1]) * math.Pi / 180
	pitch := degrees(o.ViewAngles[0]) * math.Pi / 180
	put(float64(o.Health)/100, float64(o.Armor)/100, b(o.OnGround), b(o.Ducked), float64(o.Ammo)/100, float64(o.GunFrame)/30, math.Sin(yaw), math.Cos(yaw), math.Sin(pitch), math.Cos(pitch))
	for _, x := range o.Velocity {
		put(x / 400)
	}
	put(float64(o.PreviousCommand.Forward)/400, float64(o.PreviousCommand.Side)/400, float64(o.PreviousCommand.Up)/400, b(o.PreviousCommand.Buttons&1 != 0))
	put(b(o.Weapon == "Blaster"), b(strings.Contains(o.Weapon, "/v_shotg/") || o.Weapon == "Shotgun"), b(o.InventoryAgeFrames != nil))
	if o.InventoryAgeFrames != nil {
		put(float64(*o.InventoryAgeFrames) / 30)
	} else {
		put(0)
	}
	put(b(o.Geometry != nil))
	if o.Geometry != nil {
		put(o.Geometry.UpDistance/64, b(o.Geometry.GroundDrop != nil))
		if o.Geometry.GroundDrop != nil {
			put(*o.Geometry.GroundDrop / 32)
		} else {
			put(0)
		}
	} else {
		put(0, 0, 0)
	}
	for i := 0; i < 8; i++ {
		if o.Geometry != nil && i < len(o.Geometry.Probes) {
			p := o.Geometry.Probes[i]
			put(1, p.RayDistance/256, b(p.StandingClearance != nil))
			if p.StandingClearance != nil {
				put(*p.StandingClearance / 64)
			} else {
				put(0)
			}
			put(b(p.GroundDrop != nil))
			if p.GroundDrop != nil {
				put(*p.GroundDrop / 32)
			} else {
				put(0)
			}
		} else {
			put(0, 0, 0, 0, 0, 0)
		}
	}
	local := func(x [3]float64, scale float64) {
		put((x[0]*math.Cos(yaw)+x[1]*math.Sin(yaw))/scale, (-x[0]*math.Sin(yaw)+x[1]*math.Cos(yaw))/scale, x[2]/scale)
	}
	entities := func(items []Enemy, slots int) {
		items = append([]Enemy(nil), items...)
		sort.SliceStable(items, func(i, j int) bool { return items[i].Distance < items[j].Distance })
		for i := 0; i < slots; i++ {
			if i >= len(items) {
				for j := 0; j < 12; j++ {
					put(0)
				}
				continue
			}
			e := items[i]
			put(1, e.Distance/512)
			local(e.Relative, 512)
			put(b(e.Velocity != nil))
			if e.Velocity != nil {
				local(*e.Velocity, 400)
			} else {
				put(0, 0, 0)
			}
			put(b(e.Class == "monster_parasite"), b(e.Class == "monster_gunner"), b(e.ClearShot != nil && *e.ClearShot))
		}
	}
	entities(o.Enemies, 8)
	put(b(o.Projectiles != nil))
	if o.Projectiles != nil {
		entities(*o.Projectiles, 4)
	} else {
		entities(nil, 4)
	}
	put(b(o.Teammate != nil))
	if o.Teammate != nil {
		local(*o.Teammate, 512)
	} else {
		put(0, 0, 0)
	}
	nearby := func(items *[]NearbyObject) {
		put(b(items != nil))
		objects := []NearbyObject{}
		if items != nil {
			objects = append(objects, (*items)...)
		}
		sort.SliceStable(objects, func(i, j int) bool { return objects[i].Distance < objects[j].Distance })
		for i := 0; i < 4; i++ {
			if i >= len(objects) {
				for j := 0; j < 9; j++ {
					put(0)
				}
				continue
			}
			p := objects[i]
			put(1, p.Distance/512)
			local(p.Relative, 512)
			put(b(p.Class == "misc_explobox"), b(strings.Contains(p.Class, "health")), b(p.HealthAmount != nil))
			if p.HealthAmount != nil {
				put(float64(*p.HealthAmount) / 100)
			} else {
				put(0)
			}
		}
	}
	nearby(o.Pickups)
	nearby(o.Props)
	put(b(o.Movers != nil))
	for i := 0; i < 4; i++ {
		if o.Movers != nil && i < len(*o.Movers) {
			p := (*o.Movers)[i]
			put(1)
			local(p.RelativeOrigin, 512)
			for axis := 0; axis < 3; axis++ {
				put((p.ModelMax[axis] - p.ModelMin[axis]) / 512)
			}
		} else {
			put(0, 0, 0, 0, 0, 0, 0)
		}
	}
	put(b(o.Beams != nil))
	for i := 0; i < 4; i++ {
		if o.Beams != nil && i < len(*o.Beams) {
			p := (*o.Beams)[i]
			put(1)
			local(p.StartRelative, 512)
			local(p.EndRelative, 512)
		} else {
			put(0, 0, 0, 0, 0, 0, 0)
		}
	}
	// Small history summary; no historical IDs or hidden enemy tracks.
	for i := 0; i < 4; i++ {
		if i >= len(o.History) {
			put(0, 0, 0, 0, 0, 0, 0, 0)
		} else {
			h := o.History[len(o.History)-1-i]
			put(1, float64(h.Health)/100, b(h.OnGround), b(h.Ducked))
			for _, x := range h.Velocity {
				put(x / 400)
			}
			put(b(h.PreviousCommand.Buttons&1 != 0))
		}
	}
	for i, x := range v {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, fmt.Errorf("nonfinite feature")
		}
		v[i] = math.Max(-4, math.Min(4, x))
	}
	return v, nil
}
