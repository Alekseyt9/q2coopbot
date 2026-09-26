package coord

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"syscall"
)

type Ready struct {
	Map        string `json:"map"`
	Generation int    `json:"generation"`
	Phase      int    `json:"phase"`
	Frame      int    `json:"frame"`
	Role       string `json:"role"`
}

func Read(path string) (Ready, error) {
	var r Ready
	data, err := os.ReadFile(path)
	if err != nil {
		if runtime.GOOS == "windows" && (errors.Is(err, syscall.Errno(32)) || errors.Is(err, syscall.Errno(33))) {
			return r, errors.Join(os.ErrNotExist, err)
		}
		return r, err
	}
	err = json.Unmarshal(data, &r)
	return r, err
}

func Start(a, b Ready, mapName string, generation, phase int) (int, error) {
	for _, r := range []Ready{a, b} {
		if r.Map != mapName || r.Generation != generation || r.Phase != phase || r.Frame < 1 || r.Frame > 100000 || r.Role != "actor" && r.Role != "observer" {
			return 0, fmt.Errorf("session readiness identity mismatch")
		}
	}
	if a.Role == b.Role {
		return 0, fmt.Errorf("session readiness requires both roles")
	}
	return max(a.Frame, b.Frame) + 10, nil
}
