package bot

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"q2coopbot/internal/harness/checkpoint"
)

func (c *Client) captureCheckpointRequest() error {
	if c.checkpointControl == "" {
		return nil
	}
	data, err := os.ReadFile(c.checkpointControl)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if len(data) > 4096 {
		return fmt.Errorf("checkpoint request too large")
	}
	var request checkpoint.CaptureRequest
	d := json.NewDecoder(bytes.NewReader(data))
	d.DisallowUnknownFields()
	if err = d.Decode(&request); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF || request.Version != 1 || request.ID == "" || request.Frame < 1 {
		return fmt.Errorf("invalid checkpoint capture request")
	}
	if request.ID == c.checkpointCapturedID || c.latestFrame < request.Frame || c.lastMoveFrame < c.latestFrame {
		return nil
	}
	s := c.planner.World.Snapshot
	capture := checkpoint.Capture{CaptureRequest: request, Participant: c.name, Self: s.Self, Health: s.Health}
	if s.Map != request.Map || c.spawncount != request.Generation || s.Frame != request.Frame {
		capture.Error = "client barrier anchor mismatch"
	} else {
		state, err := c.planner.CaptureCheckpoint()
		if err != nil {
			capture.Error = err.Error()
		} else {
			capture.Planner, err = json.Marshal(state)
			if err != nil {
				return err
			}
		}
		if capture.Error == "" && c.scenario != nil {
			state, err := c.scenario.CaptureCheckpoint()
			if err != nil {
				capture.Error = err.Error()
			} else {
				capture.Runner, err = json.Marshal(state)
				if err != nil {
					return err
				}
			}
		}
		if c.session != nil {
			capture.Error = "session phase checkpoint not yet supported"
		}
	}
	if err = checkpoint.WriteCapture(c.checkpointControl+".state.json", capture); err != nil {
		return err
	}
	c.checkpointCapturedID = request.ID
	return nil
}
