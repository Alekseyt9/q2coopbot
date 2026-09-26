package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"q2coopbot/internal/harness/netfault"
)

func run() error {
	path := flag.String("config", "", "path to relay JSON config")
	flag.Parse()
	if *path == "" || flag.NArg() != 0 {
		return fmt.Errorf("usage: q2netfault --config config.json")
	}
	f, err := os.Open(*path)
	if err != nil {
		return err
	}
	defer f.Close()
	var cfg struct {
		netfault.Config
		RunMS  int    `json:"run_ms"`
		Events string `json:"events"`
	}
	d := json.NewDecoder(f)
	d.DisallowUnknownFields()
	if err = d.Decode(&cfg); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("trailing config data")
	}
	if err = cfg.Config.Validate(); err != nil {
		return err
	}
	if cfg.RunMS < 1 || cfg.RunMS > 600000 || cfg.Events == "" {
		return fmt.Errorf("invalid run_ms or events path")
	}
	if cfg.ArmBarrierDir != "" && !filepath.IsAbs(cfg.ArmBarrierDir) {
		cfg.ArmBarrierDir = filepath.Join(filepath.Dir(*path), cfg.ArmBarrierDir)
	}
	output := cfg.Events
	if !filepath.IsAbs(output) {
		output = filepath.Join(filepath.Dir(*path), output)
	}
	out, err := os.OpenFile(output, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer out.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, time.Duration(cfg.RunMS)*time.Millisecond)
	defer cancel()
	encoder := json.NewEncoder(out)
	return netfault.Serve(ctx, cfg.Config, func(e netfault.Event) error { return encoder.Encode(e) }, func(addr string) { fmt.Println("relay ready:", addr) })
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
