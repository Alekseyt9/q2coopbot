package bot

import "q2coopbot/internal/quake"

func handGrenadeArmFixture(name string) bool {
	return name == "hand_grenade_armed" || name == "hand_grenade_armed_y"
}

// Explicit native fixture only: create one primed grenade, then hand control
// back to the production guard. Never fake gunframes, ammo, fuse or health.
func (c *Client) testArmHandGrenade(cmd quake.UserCmd) quake.UserCmd {
	s := c.planner.World.Snapshot
	if c.testHandGrenadeArmDone || !isHandGrenade(s.Weapon) || s.Health <= 0 {
		return cmd
	}
	if c.testHandGrenadeArmStart == 0 {
		if s.GunFrame < 16 || s.Ammo <= 0 {
			return cmd
		}
		c.testHandGrenadeArmStart = s.Frame
	}
	if s.GunFrame == 11 && c.testHandGrenadeArm11 == 0 {
		c.testHandGrenadeArm11 = s.Frame
	}
	if s.GunFrame == 11 && s.Frame-c.testHandGrenadeArm11 >= 2 || s.Frame-c.testHandGrenadeArmStart >= 20 {
		c.testHandGrenadeArmDone = true
		return cmd
	}
	cmd.Forward, cmd.Side, cmd.Up = 0, 0, 0
	cmd.Yaw, cmd.Pitch, cmd.Roll = -s.DeltaAngles[1], -s.DeltaAngles[0], -s.DeltaAngles[2]
	if c.testWeaponSwitchFixture == "hand_grenade_armed_y" {
		cmd.Yaw = int16(16384 - int(s.DeltaAngles[1]))
	}
	cmd.Buttons = 1
	c.planner.World.Command = CommandDecision{MoveSource: "none", AimSource: "test_grenade_pose", LimitReason: "test_grenade_arming"}
	return cmd
}
