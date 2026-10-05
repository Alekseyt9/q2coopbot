package bot

import "q2coopbot/internal/quake"

// A short explicit primitive exercise after the native release. Commands are
// finite and observation results are independently checked before labeling.
func teacherVerticalCommand(cmd quake.UserCmd, age int) (quake.UserCmd, bool) {
	if age < 0 || age >= 20 {
		return cmd, false
	}
	cmd.Forward, cmd.Side, cmd.Up, cmd.Buttons, cmd.Impulse = 0, 0, 0, 0, 0
	if age == 1 {
		cmd.Up = 400
	}
	if age >= 10 && age < 14 {
		cmd.Up = -400
	}
	return cmd, true
}
