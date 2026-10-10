// q2coop-proof exports validated paired native command/tick evidence.
// It does not produce trainable rollouts or bypass reset/reward validation.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"q2coopbot/internal/learningenv"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	path := flag.String("server-log", "", "Closed paired native server log")
	out := flag.String("out", "", "Fresh native paired proof JSON")
	flag.Parse()
	if *path == "" || *out == "" {
		return fmt.Errorf("server-log and out required")
	}
	data, err := os.ReadFile(*path)
	if err != nil {
		return err
	}
	events, err := learningenv.ReadDamageEvents(bytes.NewReader(data))
	if err != nil {
		return err
	}
	pairs, err := learningenv.ReadPairedNativeSteps(bytes.NewReader(data), events)
	if err != nil {
		return err
	}
	digest := sha256.Sum256(data)
	result := struct {
		Version   string                   `json:"version"`
		Source    string                   `json:"source"`
		SHA       string                   `json:"source_sha256"`
		Trainable bool                     `json:"ppo_trainable"`
		Native    *learningenv.NativePairs `json:"native"`
		Scope     string                   `json:"scope"`
	}{"coop_native_pair_proof_v1", *path, hex.EncodeToString(digest[:]), false, pairs, "Paired commands, world steps and shared effect indexes only. Reset, per-actor reward, terminal and both-client trace validation remain required."}
	f, err := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(f)
	err = encoder.Encode(result)
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	fmt.Printf("Verified %d native pairs, seed %d; PPO eligibility false\n", len(pairs.Pairs), pairs.Release.Seed)
	return nil
}
