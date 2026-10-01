package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"q2coopbot/internal/bot"
	"q2coopbot/internal/quake"
)

func main() {
	trace := flag.String("trace", "", "native bot JSONL")
	root := flag.String("root", "", "baseq2 assets")
	name := flag.String("map", "base1", "map")
	out := flag.String("out", "", "report JSON")
	contact := flag.Bool("contact", false, "stationary native damageable contact fixture")
	movingContact := flag.Bool("moving-contact", false, "walking native damageable contact fixture")
	flag.Parse()
	if *contact && *movingContact {
		fmt.Fprintln(os.Stderr, "select one contact fixture")
		os.Exit(1)
	}
	if err := run(*trace, *root, *name, *out, *contact, *movingContact); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(trace, root, name, out string, contact, movingContact bool) error {
	f, err := os.Open(trace)
	if err != nil {
		return err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 4<<20)
	rows := []quake.Snapshot{}
	for s.Scan() {
		var row quake.Snapshot
		if err = json.Unmarshal(s.Bytes(), &row); err != nil {
			return err
		}
		rows = append(rows, row)
	}
	if err = s.Err(); err != nil {
		return err
	}
	g, err := quake.LoadMap(root, name)
	if err != nil {
		return err
	}
	var r bot.GrenadeCalibration
	var validation error
	if movingContact {
		r, validation = bot.CheckGrenadeMovingContact(rows, &g)
	} else if contact {
		r, validation = bot.CheckGrenadeContact(rows, &g)
	} else {
		r, validation = bot.CalibrateGrenade(rows, &g)
	}
	data, err := json.MarshalIndent(struct {
		Accepted bool                   `json:"accepted"`
		Report   bot.GrenadeCalibration `json:"report"`
		Reason   string                 `json:"reason"`
	}{validation == nil, r, fmt.Sprint(validation)}, "", "  ")
	if err != nil {
		return err
	}
	if err = os.WriteFile(out, data, 0600); err != nil {
		return err
	}
	return validation
}
