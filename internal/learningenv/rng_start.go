package learningenv

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

var rngStartPattern = regexp.MustCompile(`^g_test_rng_start game_frame=(\d+) phase=post_frame seed=(\d+) cursor_before=(\d+) cursor_after=256$`)
var weaponStartPattern = regexp.MustCompile(`^g_test_weapon_start game_frame=(\d+) actor=1 weapon=(Blaster|Machinegun) gunframe_before=(\d+) gunframe_after=(\d+)$`)

// VerifyPostFrameRNG verifies the native post-frame seeding receipt. The cursor
// is a diagnostic, not proof of complete RNG/world snapshot equivalence.
func VerifyPostFrameRNG(r io.Reader, seed, expectedFrame int, expectedWeapon ...string) error {
	weapon := "Blaster"
	if len(expectedWeapon) > 1 {
		return fmt.Errorf("one fixed weapon expected")
	}
	if len(expectedWeapon) == 1 {
		weapon = expectedWeapon[0]
	}
	minFrame, maxFrame := 9, 52
	if weapon == "Machinegun" {
		minFrame, maxFrame = 6, 45
	} else if weapon != "Blaster" {
		return fmt.Errorf("unsupported fixed weapon")
	}
	s := bufio.NewScanner(r)
	count, releases, frame, releaseFrame := 0, 0, 0, 0
	weapons := 0
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if strings.HasPrefix(line, "g_test_weapon_start ") {
			m := weaponStartPattern.FindStringSubmatch(line)
			if m == nil {
				return fmt.Errorf("malformed weapon start receipt")
			}
			f, err := strconv.Atoi(m[1])
			if err != nil {
				return err
			}
			g, err := strconv.Atoi(m[3])
			if err != nil {
				return err
			}
			after, err := strconv.Atoi(m[4])
			if err != nil {
				return err
			}
			if expectedFrame == 0 || f != expectedFrame || m[2] != weapon || g < minFrame || g > maxFrame || after != minFrame {
				return fmt.Errorf("weapon start differs")
			}
			weapons++
		}
		if strings.HasPrefix(line, "g_test_rng_start ") {
			m := rngStartPattern.FindStringSubmatch(line)
			if m == nil {
				return fmt.Errorf("malformed post-frame RNG receipt")
			}
			values := make([]int, 3)
			for i := range values {
				v, err := strconv.Atoi(m[i+1])
				if err != nil {
					return err
				}
				values[i] = v
			}
			if values[1] != seed || values[2] >= 0x200000 {
				return fmt.Errorf("RNG receipt seed/cursor mismatch")
			}
			frame = values[0]
			count++
		}
		if m := releasePattern.FindStringSubmatch(line); m != nil {
			v, err := strconv.Atoi(m[3])
			if err != nil {
				return err
			}
			releaseFrame = v
			v, err = strconv.Atoi(m[4])
			if err != nil {
				return err
			}
			if v != seed {
				return fmt.Errorf("RNG release seed mismatch")
			}
			releases++
		}
	}
	if err := s.Err(); err != nil {
		return err
	}
	if count != 1 || releases != 1 || frame != releaseFrame || expectedFrame > 0 && (frame != expectedFrame || weapons != 1) {
		return fmt.Errorf("post-frame RNG release unconfirmed")
	}
	return nil
}
