package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"

	"q2coopbot/internal/bot"
)

func main() {
	configPath := flag.String("config", "", "path to the Go companion JSON config")
	flag.Parse()
	if *configPath == "" || flag.NArg() != 0 {
		log.Fatal("usage: q2coopbot --config path/to/config.json")
	}
	cfg, err := bot.LoadConfig(*configPath)
	if err != nil {
		log.Fatal(err)
	}
	cfg.TestRCONPassword = os.Getenv("Q2COOPBOT_TEST_RCON")
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := bot.Run(ctx, cfg); err != nil {
		log.Fatal(err)
	}
}
