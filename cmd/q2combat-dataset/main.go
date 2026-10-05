package main

import (
	"flag"
	"fmt"
	"os"

	"q2coopbot/internal/demodata"
)

type batches []string

func (b *batches) String() string         { return fmt.Sprint([]string(*b)) }
func (b *batches) Set(value string) error { *b = append(*b, value); return nil }
func main() {
	spec := flag.String("spec", "", "Frozen seed/split specification")
	out := flag.String("out", "", "Fresh dataset directory")
	var roots batches
	flag.Var(&roots, "batch", "Verified baseline batch (repeatable)")
	flag.Parse()
	if *spec == "" || *out == "" || len(roots) == 0 {
		fmt.Fprintln(os.Stderr, "spec, out and batch required")
		os.Exit(1)
	}
	report, err := demodata.Build(*spec, roots, *out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("Dataset ready: train=%d validation=%d test=%d\n", report.Splits["train"].Candidates, report.Splits["validation"].Candidates, report.Splits["test"].Candidates)
}
