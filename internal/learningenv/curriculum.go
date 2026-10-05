package learningenv

import (
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
)

var curriculumPattern = regexp.MustCompile(`^g_test_curriculum monster_health map=base1 game_frame=(\d+) entity=(\d+) class=monster_parasite before=175 after=(\d+)$`)

// VerifyCurriculum checks offline initialization evidence; health is never a
// policy feature. Ordinary episodes must contain no curriculum override.
func VerifyCurriculum(r io.Reader, health, seed int) (int, error) {
	if health != 0 && health != 10 && health != 20 && health != 30 && health != 40 && health != 60 && health != 100 {
		return 0, fmt.Errorf("unsupported curriculum health")
	}
	s := bufio.NewScanner(r)
	count, target, frame, releaseFrame, releaseSeed, releases := 0, 0, 0, 0, 0, 0
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if strings.HasPrefix(line, "g_test_curriculum ") {
			m := curriculumPattern.FindStringSubmatch(line)
			if m == nil {
				return 0, fmt.Errorf("malformed curriculum initialization")
			}
			count++
			frame, _ = strconv.Atoi(m[1])
			target, _ = strconv.Atoi(m[2])
			hp, _ := strconv.Atoi(m[3])
			if hp != health || target < 1 {
				return 0, fmt.Errorf("curriculum health differs")
			}
		}
		if m := releasePattern.FindStringSubmatch(line); m != nil {
			releases++
			releaseFrame, _ = strconv.Atoi(m[3])
			releaseSeed, _ = strconv.Atoi(m[4])
		}
	}
	if err := s.Err(); err != nil {
		return 0, err
	}
	if health == 0 && count == 0 {
		return 0, nil
	}
	if count != 1 || releases != 1 || frame != releaseFrame || releaseSeed != seed {
		return 0, fmt.Errorf("curriculum release not confirmed")
	}
	return target, nil
}
