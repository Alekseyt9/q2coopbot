package checkpoint

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"q2coopbot/internal/quake"
)

type Barrier struct {
	ID         string   `json:"id"`
	Map        string   `json:"map"`
	Frame      int      `json:"frame"`
	Generation int      `json:"generation"`
	Controls   []string `json:"controls"`
}
type CaptureRequest struct {
	Version    int    `json:"version"`
	ID         string `json:"id"`
	Map        string `json:"map"`
	Frame      int    `json:"frame"`
	Generation int    `json:"generation"`
}
type Capture struct {
	CaptureRequest
	Participant string          `json:"participant"`
	Error       string          `json:"error,omitempty"`
	Planner     json.RawMessage `json:"planner,omitempty"`
	Runner      json.RawMessage `json:"runner,omitempty"`
	Self        quake.Vec3      `json:"self"`
	Health      int16           `json:"health"`
	SelfEntity  int             `json:"self_entity,omitempty"`
}
type BarrierProof struct {
	ID           string `json:"id"`
	Frame        int    `json:"frame"`
	Generation   int    `json:"generation"`
	Participants []File `json:"participants"`
}

func WriteCapture(path string, value any) error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(path+".tmp", data, 0600); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

func (b Barrier) validate() error {
	if !identifier.MatchString(b.ID) || !identifier.MatchString(b.Map) || b.Frame < 1 || b.Frame > 1000000 || b.Generation < 1 || len(b.Controls) < 1 || len(b.Controls) > 8 {
		return fmt.Errorf("invalid checkpoint barrier")
	}
	seen := map[string]bool{}
	for _, path := range b.Controls {
		path = filepath.Clean(path)
		if path == "." || seen[strings.ToLower(path)] {
			return fmt.Errorf("invalid/duplicate checkpoint control")
		}
		seen[strings.ToLower(path)] = true
	}
	return nil
}

func validateCapture(c Capture, r CaptureRequest) error {
	if c.CaptureRequest != r || c.Error != "" || !identifier.MatchString(c.Participant) || c.Health <= 0 || !json.Valid(c.Planner) {
		return fmt.Errorf("checkpoint participant identity/state rejected")
	}
	var anchor struct {
		Version int    `json:"version"`
		Map     string `json:"map"`
		Frame   int    `json:"captured_frame"`
	}
	if json.Unmarshal(c.Planner, &anchor) != nil || anchor.Version != 1 || anchor.Map != r.Map || anchor.Frame != r.Frame {
		return fmt.Errorf("planner capture frame mismatch")
	}
	if len(c.Runner) > 0 {
		var runner struct {
			Version int `json:"version"`
			Frame   int `json:"captured_frame"`
		}
		if json.Unmarshal(c.Runner, &runner) != nil || runner.Version != 1 || runner.Frame != r.Frame {
			return fmt.Errorf("runner capture frame mismatch")
		}
	}
	return nil
}

func collectBarrier(ctx context.Context, c Config, password string) ([]Capture, error) {
	b := c.Barrier
	timeout := time.Duration(c.TimeoutMS) * time.Millisecond
	r := CaptureRequest{Version: 1, ID: b.ID, Map: b.Map, Frame: b.Frame, Generation: b.Generation}
	for _, path := range b.Controls {
		if err := WriteCapture(path, r); err != nil {
			return nil, err
		}
	}
	if _, err := request(ctx, c.Server, password, fmt.Sprintf("set sv_test_checkpoint_frame %d", b.Frame), timeout); err != nil {
		return nil, err
	}
	deadline := time.Now().Add(timeout)
	var captures []Capture
	for {
		captures = nil
		names := map[string]bool{}
		ready := true
		for _, path := range b.Controls {
			var capture Capture
			err := readJSON(path+".state.json", &capture)
			if os.IsNotExist(err) || runtime.GOOS == "windows" && (errors.Is(err, syscall.Errno(32)) || errors.Is(err, syscall.Errno(33))) {
				ready = false
				break
			}
			if err != nil {
				return nil, err
			}
			if capture.ID != r.ID {
				ready = false
				break
			}
			if err = validateCapture(capture, r); err != nil {
				return nil, err
			}
			if names[strings.ToLower(capture.Participant)] {
				return nil, fmt.Errorf("duplicate checkpoint participant")
			}
			names[strings.ToLower(capture.Participant)] = true
			captures = append(captures, capture)
		}
		if ready {
			break
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("checkpoint participant timeout")
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(20 * time.Millisecond):
		}
	}
	if err := verifyBarrier(ctx, c, password); err != nil {
		return nil, err
	}
	return captures, nil
}
func verifyBarrier(ctx context.Context, c Config, password string) error {
	armed, err := request(ctx, c.Server, password, "sv_test_checkpoint_frame", time.Duration(c.TimeoutMS)*time.Millisecond)
	if err != nil {
		return err
	}
	if !strings.Contains(armed, fmt.Sprintf(`"sv_test_checkpoint_frame" is "%d"`, c.Barrier.Frame)) {
		return fmt.Errorf("native checkpoint barrier is not armed")
	}
	reply, err := request(ctx, c.Server, password, "sv_test_checkpoint_ack", time.Duration(c.TimeoutMS)*time.Millisecond)
	if err != nil {
		return err
	}
	want := fmt.Sprintf(`"sv_test_checkpoint_ack" is "%d:%d:%s"`, c.Barrier.Generation, c.Barrier.Frame, c.Barrier.Map)
	if !strings.Contains(reply, want) {
		return fmt.Errorf("native checkpoint barrier anchor mismatch")
	}
	return nil
}

func saveCaptures(dir string, b *Barrier, captures []Capture) (*BarrierProof, error) {
	root := filepath.Join(dir, "sidecar")
	if err := os.Mkdir(root, 0755); err != nil {
		return nil, err
	}
	proof := &BarrierProof{ID: b.ID, Frame: b.Frame, Generation: b.Generation}
	for _, capture := range captures {
		name := capture.Participant + ".json"
		if err := WriteCapture(filepath.Join(root, name), capture); err != nil {
			return nil, err
		}
		record, err := fileRecord(root, name)
		if err != nil {
			return nil, err
		}
		proof.Participants = append(proof.Participants, record)
	}
	return proof, nil
}

func verifySavedCaptures(dir, mapName string, proof *BarrierProof) error {
	request := CaptureRequest{Version: 1, ID: proof.ID, Map: mapName, Frame: proof.Frame, Generation: proof.Generation}
	if !identifier.MatchString(proof.ID) || proof.Frame < 1 || proof.Generation < 1 || len(proof.Participants) < 1 || len(proof.Participants) > 8 {
		return fmt.Errorf("invalid saved barrier")
	}
	seen := map[string]bool{}
	root := filepath.Join(dir, "sidecar")
	for _, record := range proof.Participants {
		name := strings.TrimSuffix(record.Name, ".json")
		if !identifier.MatchString(name) || !strings.HasSuffix(record.Name, ".json") || seen[strings.ToLower(name)] {
			return fmt.Errorf("invalid/duplicate sidecar file name")
		}
		seen[strings.ToLower(name)] = true
		actual, err := fileRecord(root, record.Name)
		if err != nil || actual != record {
			return fmt.Errorf("checkpoint sidecar integrity mismatch")
		}
		var capture Capture
		if err = readJSON(filepath.Join(root, record.Name), &capture); err != nil {
			return err
		}
		if capture.Participant != name {
			return fmt.Errorf("sidecar participant differs from file name")
		}
		if err = validateCapture(capture, request); err != nil {
			return err
		}
	}
	return nil
}

func participantBindings(dir string, proof *BarrierProof) (map[int]string, error) {
	if proof == nil || len(proof.Participants) == 0 {
		return nil, fmt.Errorf("participant binding requires a coordinated checkpoint")
	}
	bindings := map[int]string{}
	for _, record := range proof.Participants {
		var capture Capture
		if err := readJSON(filepath.Join(dir, "sidecar", record.Name), &capture); err != nil {
			return nil, err
		}
		// svs.clients supports at most 256 player slots; names are stored in
		// the native client as at most 31 bytes. No lossy identity is allowed.
		if capture.SelfEntity < 1 || capture.SelfEntity > 256 || len(capture.Participant) > 31 || bindings[capture.SelfEntity] != "" {
			return nil, fmt.Errorf("invalid/duplicate native participant slot")
		}
		bindings[capture.SelfEntity] = capture.Participant
	}
	return bindings, nil
}

// ReadParticipant verifies the complete sidecar set before exposing one payload.
// Native loading and runtime identity are checked separately by the controller.
func ReadParticipant(dir, name string) (Capture, error) {
	var manifest Manifest
	var capture Capture
	if !identifier.MatchString(name) {
		return capture, fmt.Errorf("invalid participant name")
	}
	if err := readJSON(filepath.Join(dir, "manifest.json"), &manifest); err != nil {
		return capture, err
	}
	if manifest.Version != 1 || manifest.Barrier == nil {
		return capture, fmt.Errorf("restore requires coordinated checkpoint")
	}
	if err := verifySavedCaptures(dir, manifest.Map, manifest.Barrier); err != nil {
		return capture, err
	}
	if _, err := participantBindings(dir, manifest.Barrier); err != nil {
		return capture, err
	}
	for _, record := range manifest.Barrier.Participants {
		if record.Name == name+".json" {
			err := readJSON(filepath.Join(dir, "sidecar", record.Name), &capture)
			return capture, err
		}
	}
	return capture, fmt.Errorf("checkpoint participant missing")
}
