// q2collisiondata extracts observed protocol collision bodies from a DM2.
// It does not recover user input or execute a learned policy.
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"q2coopbot/internal/quake"
)

func main() {
	demo := flag.String("demo", "", "source DM2")
	out := flag.String("out", "", "fresh observed-body JSONL")
	flag.Parse()
	if err := extract(*demo, *out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func extract(demo, out string) error {
	if demo == "" || out == "" {
		return fmt.Errorf("demo and out required")
	}
	in, err := os.Open(demo)
	if err != nil {
		return err
	}
	defer in.Close()
	f, err := os.OpenFile(out, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	hash := sha256.New()
	reader := io.TeeReader(in, hash)
	d := quake.NewDecoder()
	encoder := json.NewEncoder(f)
	count := 0
	for {
		var header [4]byte
		if _, err = io.ReadFull(reader, header[:]); err != nil {
			return fmt.Errorf("missing DM2 terminator: %w", err)
		}
		n := int32(binary.LittleEndian.Uint32(header[:]))
		if n == -1 {
			break
		}
		if n < 0 || n > 2*1024*1024 {
			return fmt.Errorf("invalid packet size %d", n)
		}
		payload := make([]byte, n)
		if _, err = io.ReadFull(reader, payload); err != nil {
			return err
		}
		frames, err := d.Parse(payload)
		if err != nil {
			return err
		}
		for _, frame := range frames {
			s := d.Snapshot(frame)
			if err = encoder.Encode(struct {
				Map      string         `json:"map"`
				Frame    int            `json:"frame"`
				Position quake.Vec3     `json:"position"`
				Bodies   []quake.Object `json:"bodies"`
			}{s.Map, frame.Number, s.Self, s.CollisionBodies}); err != nil {
				return err
			}
			count++
		}
	}
	var extra [1]byte
	if n, _ := reader.Read(extra[:]); n != 0 {
		return fmt.Errorf("trailing data after terminator")
	}
	if err = f.Close(); err != nil {
		return err
	}
	report := struct {
		State           string `json:"state"`
		DemoSHA256      string `json:"demo_sha256"`
		Frames          int    `json:"frames"`
		NeuralExecution bool   `json:"neural_execution"`
	}{"complete", hex.EncodeToString(hash.Sum(nil)), count, false}
	return json.NewEncoder(os.Stdout).Encode(report)
}
