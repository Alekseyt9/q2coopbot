package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"q2coopbot/internal/learningenv"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

func main(){if err:=run();err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(1)}}

func run() error {
	input:=flag.String("trace","","Client trace JSONL")
	out:=flag.String("out","","Fresh output directory")
	worker:=flag.String("worker","","Worker identity")
	episode:=flag.String("episode","","Episode identity")
	flag.Parse()
	if *input=="" || *out=="" || *worker=="" || *episode=="" {return fmt.Errorf("trace, out, worker and episode required")}
	if err:=os.Mkdir(*out,0755);err!=nil{return err}
	f,err:=os.Open(*input);if err!=nil{return err};defer f.Close()
	steps,err:=os.Create(filepath.Join(*out,"steps.jsonl"));if err!=nil{return err};defer steps.Close()
	outcomes,err:=os.Create(filepath.Join(*out,"outcomes.jsonl"));if err!=nil{return err};defer outcomes.Close()
	stepWriter,outcomeWriter:=bufio.NewWriter(steps),bufio.NewWriter(outcomes)
	se,oe:=json.NewEncoder(stepWriter),json.NewEncoder(outcomeWriter)
	a:=learningenv.Assembler{Worker:*worker,Episode:*episode}
	count,terminals,truncated:=0,0,0
	emit:=func(s *learningenv.Step,o *learningenv.Outcome) error {
		if s==nil{return nil}
		if err:=se.Encode(s);err!=nil{return err};if err:=oe.Encode(o);err!=nil{return err}
		count++;if s.Terminal{terminals++};if s.Truncated{truncated++};return nil
	}
	scanner:=bufio.NewScanner(f);scanner.Buffer(make([]byte,65536),8*1024*1024)
	line:=0
	for scanner.Scan(){
		line++
		var row struct {Capture *policy.Capture `json:"combat_policy"`;Command quake.UserCmd `json:"sent_command"`}
		if err:=json.Unmarshal(scanner.Bytes(),&row);err!=nil{return fmt.Errorf("line %d: %w",line,err)}
		if row.Capture==nil || row.Capture.AppliedCommand!=row.Command{return fmt.Errorf("line %d: missing/mismatched capture",line)}
		s,o,err:=a.Push(*row.Capture);if err!=nil{return fmt.Errorf("line %d: %w",line,err)}
		if err:=emit(s,o);err!=nil{return err}
	}
	if err:=scanner.Err();err!=nil{return err}
	s,o:=a.Close("client_time_limit_or_trace_end");if err:=emit(s,o);err!=nil{return err}
	if err:=stepWriter.Flush();err!=nil{return err};if err:=outcomeWriter.Flush();err!=nil{return err}
	if count==0{return fmt.Errorf("no usable steps")}
	summary:=struct{Version string `json:"version"`;Steps int `json:"steps"`;Terminals int `json:"terminals"`;Truncated int `json:"truncated"`;Scope string `json:"scope"`}{learningenv.StepVersion,count,terminals,truncated,"Observed transitions only; sent commands are not per-command server acknowledgements. No victory or scalar reward inferred."}
	data,err:=json.MarshalIndent(summary,"","  ");if err!=nil{return err}
	return os.WriteFile(filepath.Join(*out,"report.json"),data,0644)
}
