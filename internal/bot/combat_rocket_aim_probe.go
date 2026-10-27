package bot

import (
	"fmt"
	"math"

	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"q2coopbot/internal/rocketpractice"
)

func validateRocketAimProbe(cfg Config, contract string) (*quake.Vec3, error) {
	frame := cfg.TestRocketAimProbeFrame
	point := cfg.TestRocketAimProbePoint
	if frame == 0 && point == "" {
		return nil, nil
	}
	if frame <= 0 || frame > 1000000 || point == "" || !cfg.TestSynchronous || !cfg.TestCombatOnly || cfg.CombatMode != "learned" || !cfg.CombatCapture || cfg.Host != "127.0.0.1" || !cfg.CombatCheckedWallSlide || contract != policy.StockWeaponsFireCombatExecution || cfg.CombatActionQualityGuard || cfg.CombatDiagnosticNominalAim || cfg.TestCombatAdvice || cfg.TestGrenadeReleaseProbeFrame != 0 || cfg.TestForcedDuckProbeFrame != 0 || cfg.TestStoppedCrouchProbeFrame != 0 || cfg.TestStoppedCrouchProbeAllFrames || cfg.TestCheckedGroundRequestProbeFrame != 0 || cfg.TestCheckedGroundRequestProbeAllFrames {
		return nil, fmt.Errorf("rocket aim probe requires exclusive captured local synchronous stock-weapon learned fixture and point/frame")
	}
	v, err := parseTestTeleport(point)
	if err != nil {
		return nil, fmt.Errorf("rocket aim probe point: %w", err)
	}
	return &v, nil
}

// First acquisition protocol requires zero horizontal motion: changing view
// cannot rotate a movement request. No attack, weapon or vertical bit is added.
func diagnosticRocketSurfaceAim(o policy.Observation, s quake.Snapshot, world *quake.MapInfo, cmd quake.UserCmd, point quake.Vec3) (quake.UserCmd, []policy.Intervention) {
	if s.Health <= 0 || s.Ammo <= 0 || (s.Weapon != "Rocket Launcher" && s.Weapon != "models/weapons/v_rocket/tris.md2") || s.GunFrame < 13 || cmd.Buttons&1 == 0 || cmd.Forward != 0 || cmd.Side != 0 {
		return cmd, nil
	}
	options, err := rocketpractice.Generate(o, world)
	if err != nil {
		return cmd, nil
	}
	known := false
	for _, candidate := range options {
		if quake.Distance(candidate.Point, point) < .01 {
			known = true
			break
		}
	}
	if !known {
		return cmd, nil
	}
	eye := s.EyePoint()
	dx, dy, dz := point[0]-eye[0], point[1]-eye[1], point[2]-eye[2]
	if math.Hypot(math.Hypot(dx, dy), dz) < 1 {
		return cmd, nil
	}
	pitch := -math.Atan2(dz, math.Hypot(dx, dy)) * 180 / math.Pi
	yaw := math.Atan2(dy, dx) * 180 / math.Pi
	pitch = math.Max(-89, math.Min(89, pitch))
	before := cmd
	cmd.Pitch = int16(int(math.Round(pitch*65536/360))) - s.DeltaAngles[0]
	cmd.Yaw = int16(int(math.Round(yaw*65536/360))) - s.DeltaAngles[1]
	changes := []policy.Intervention{}
	if cmd.Pitch != before.Pitch {
		changes = append(changes, policy.Intervention{Component: "pitch", Reason: "diagnostic_rocket_surface_aim"})
	}
	if cmd.Yaw != before.Yaw {
		changes = append(changes, policy.Intervention{Component: "yaw", Reason: "diagnostic_rocket_surface_aim"})
	}
	return cmd, changes
}
