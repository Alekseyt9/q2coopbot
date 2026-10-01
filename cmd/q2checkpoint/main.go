package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"q2coopbot/internal/harness/checkpoint"
)

func main() {
	path := flag.String("config", "", "checkpoint operation JSON")
	flag.Parse()
	if *path == "" || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: q2checkpoint --config file.json")
		os.Exit(2)
	}
	c, err := checkpoint.LoadConfig(*path)
	if err == nil {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		var result checkpoint.Result
		result, err = checkpoint.Run(ctx, c, os.Getenv("Q2COOPBOT_TEST_RCON"))
		if err == nil {
			err = json.NewEncoder(os.Stdout).Encode(result)
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
