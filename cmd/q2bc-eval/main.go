package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"q2coopbot/internal/policy"
	"sort"
	"time"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	weights := flag.String("weights", "", "weights JSON")
	reference := flag.String("reference", "", "PyTorch reference JSONL")
	out := flag.String("out", "", "fresh report")
	flag.Parse()
	p, e := policy.LoadMLP(*weights)
	if e != nil {
		return e
	}
	f, e := os.Open(*reference)
	if e != nil {
		return e
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 65536), 2*1024*1024)
	count := 0
	maxError := 0.0
	times := []float64{}
	for scanner.Scan() {
		var r struct {
			Observation policy.Observation `json:"observation"`
			Raw         []float64          `json:"raw"`
		}
		if e = json.Unmarshal(scanner.Bytes(), &r); e != nil {
			return e
		}
		x, e := p.Raw(r.Observation)
		if e != nil {
			return e
		}
		// Batch to exceed coarse Windows timer resolution. Include action checks.
		start := time.Now()
		for i := 0; i < 32; i++ {
			if _, e = p.Decide(r.Observation); e != nil {
				return e
			}
		}
		times = append(times, float64(time.Since(start).Nanoseconds())/32000)
		if len(x) != len(r.Raw) {
			return fmt.Errorf("raw dimensions")
		}
		for i := range x {
			maxError = math.Max(maxError, math.Abs(x[i]-r.Raw[i]))
		}
		if _, e = p.Decide(r.Observation); e != nil {
			return e
		}
		count++
	}
	if e = scanner.Err(); e != nil {
		return e
	}
	if count == 0 || maxError > 1e-4 {
		return fmt.Errorf("Go/PyTorch parity failed %.9g", maxError)
	}
	sort.Float64s(times)
	report := map[string]any{"rows": count, "max_raw_error": maxError, "p50_us": times[len(times)/2], "p95_us": times[int(float64(len(times)-1)*.95)], "p99_us": times[int(float64(len(times)-1)*.99)], "max_us": times[len(times)-1], "latency_batch": 32, "policy_version": p.Version(), "scope": "Offline forward parity and amortized Decide latency, not single-call tail or gameplay acceptance"}
	data, _ := json.MarshalIndent(report, "", "  ")
	fmt.Println(string(data))
	output, e := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if e != nil {
		return e
	}
	defer output.Close()
	_, e = output.Write(data)
	return e
}
