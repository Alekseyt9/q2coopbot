package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"q2coopbot/internal/harness/netfault"
)

func run() error {
	path:=flag.String("config","","report config JSON");flag.Parse()
	if *path==""||flag.NArg()!=0{return fmt.Errorf("usage: q2netfault-report --config config.json")}
	f,err:=os.Open(*path);if err!=nil{return err};defer f.Close()
	var cfg struct{Network netfault.Config `json:"network"`;Events string `json:"events"`;Output string `json:"output"`;RecoveryMS int `json:"recovery_ms"`;RequireUpstreamLoss bool `json:"require_upstream_loss"`}
	d:=json.NewDecoder(f);d.DisallowUnknownFields();if err=d.Decode(&cfg);err!=nil{return err};if d.Decode(new(any))!=io.EOF{return fmt.Errorf("trailing config")}
	if cfg.Events==""||cfg.Output==""{return fmt.Errorf("events and output required")}
	resolve:=func(p string)string{if filepath.IsAbs(p){return filepath.Clean(p)};return filepath.Join(filepath.Dir(*path),p)}
	if resolve(cfg.Events)==resolve(cfg.Output){return fmt.Errorf("output cannot replace event log")}
	rows,err:=netfault.ReadEvents(resolve(cfg.Events))
	var r netfault.Report
	if err!=nil{r=netfault.Report{State:"trace_invalid",Reason:err.Error()}}else{r=netfault.Analyze(cfg.Network,rows,cfg.RecoveryMS,cfg.RequireUpstreamLoss)}
	data,err:=json.MarshalIndent(r,"","  ");if err!=nil{return err}
	if err=os.WriteFile(resolve(cfg.Output),data,0600);err!=nil{return err}
	if !r.Accepted{return fmt.Errorf("network report rejected: %s",r.Reason)}
	fmt.Println("Network transport accepted:",resolve(cfg.Output));return nil
}
func main(){if err:=run();err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(1)}}
