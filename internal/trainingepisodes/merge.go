package trainingepisodes

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
)

// MergeRollouts joins fresh exports from a single behavior policy. All native
// input receipts remain pinned; sequences and seed identities stay separate.
func MergeRollouts(dirs []string, out string) error {
	if len(dirs) < 1 {
		return fmt.Errorf("select fresh rollout directories")
	}
	sources := map[string]string{}
	owners := map[int]int{}
	var reference map[string]any
	var rollout, context bytes.Buffer
	rows, contexts, skipped, terminals := 0, 0, 0, 0
	members := []map[string]any{}
	for i, dir := range dirs {
		dir, err := filepath.Abs(dir)
		if err != nil {
			return err
		}
		reportPath := filepath.Join(dir, "report.json")
		var meta map[string]any
		if err := read(reportPath, &meta); err != nil {
			return err
		}
		if meta["version"] != "combat_ppo_rollout_v1" {
			return fmt.Errorf("unsupported rollout")
		}
		keys := []string{"version", "feature_version", "reward_config_sha256", "reward_version", "aim_gamma", "training_monster_health", "policy_version", "model_sha256", "recurrent_version", "numerical_verification"}
		if reference == nil {
			reference = meta
		} else {
			for _, k := range keys {
				if !reflect.DeepEqual(reference[k], meta[k]) {
					return fmt.Errorf("mixed behavior or objective: %s", k)
				}
			}
		}
		rawSources, ok := meta["source_sha256"].(map[string]any)
		if !ok || len(rawSources) == 0 {
			return fmt.Errorf("missing native receipts")
		}
		for path, value := range rawSources {
			digest, ok := value.(string)
			if !ok {
				return fmt.Errorf("invalid receipt")
			}
			if old, exists := sources[path]; exists && old != digest {
				return fmt.Errorf("conflicting source receipt")
			}
			sources[path] = digest
		}
		for _, file := range []string{"report.json", "rollout.jsonl"} {
			path := filepath.Join(dir, file)
			h, err := Hash(path)
			if err != nil {
				return err
			}
			sources[path] = h
		}
		if sources[filepath.Join(dir, "rollout.jsonl")] != meta["rollout_sha256"] {
			return fmt.Errorf("rollout content changed")
		}
		data, err := os.ReadFile(filepath.Join(dir, "rollout.jsonl"))
		if err != nil {
			return err
		}
		n, err := checkMergedRows(data, i, owners)
		if err != nil {
			return err
		}
		if float64(n) != meta["rows"] {
			return fmt.Errorf("row count differs")
		}
		rows += n
		rollout.Write(data)
		if n > 0 && len(data) > 0 && data[len(data)-1] != '\n' {
			rollout.WriteByte('\n')
		}
		if meta["recurrent_version"] != nil {
			path := filepath.Join(dir, "sequence.jsonl")
			h, err := Hash(path)
			if err != nil {
				return err
			}
			if h != meta["sequence_sha256"] {
				return fmt.Errorf("sequence content changed")
			}
			sources[path] = h
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			n, err := checkMergedRows(data, i, owners)
			if err != nil {
				return err
			}
			if float64(n) != meta["sequence_rows"] {
				return fmt.Errorf("context count differs")
			}
			contexts += n
			context.Write(data)
			if n > 0 && data[len(data)-1] != '\n' {
				context.WriteByte('\n')
			}
		}
		s, _ := meta["skipped"].(float64)
		t, _ := meta["terminals"].(float64)
		skipped += int(s)
		terminals += int(t)
		members = append(members, map[string]any{"directory": dir, "rows": meta["rows"], "rollout_sha256": meta["rollout_sha256"]})
	}
	for path, h := range sources {
		actual, err := Hash(path)
		if err != nil || actual != h {
			return fmt.Errorf("source changed: %s", path)
		}
	}
	if rows < 2 {
		return fmt.Errorf("insufficient verified transitions")
	}
	if err := os.Mkdir(out, 0755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "rollout.jsonl"), rollout.Bytes(), 0644); err != nil {
		return err
	}
	reference["rows"] = rows
	reference["skipped"] = skipped
	reference["terminals"] = terminals
	reference["source_sha256"] = sources
	reference["members"] = members
	// Per-case CUDA receipts are pinned in source_sha256; a merged corpus has
	// no single per-case verification file representing every member.
	delete(reference, "cuda_verification_sha256")
	reference["scope"] = "Joint fresh on-policy curriculum batch; all native receipts pinned; per-episode seeds and memory context retained. Sample share follows eligible transitions."
	h, err := Hash(filepath.Join(out, "rollout.jsonl"))
	if err != nil {
		return err
	}
	reference["rollout_sha256"] = h
	if reference["recurrent_version"] != nil {
		path := filepath.Join(out, "sequence.jsonl")
		if err := os.WriteFile(path, context.Bytes(), 0644); err != nil {
			return err
		}
		h, err := Hash(path)
		if err != nil {
			return err
		}
		reference["sequence_sha256"] = h
		reference["sequence_rows"] = contexts
	}
	b, err := json.MarshalIndent(reference, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(out, "report.json"), b, 0644)
}
func checkMergedRows(data []byte, owner int, owners map[int]int) (int, error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 65536), 4*1024*1024)
	n := 0
	seen := map[[2]int]bool{}
	for scanner.Scan() {
		var v struct {
			Seed  *int `json:"seed"`
			Index *int `json:"index"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &v); err != nil {
			return 0, err
		}
		if v.Seed == nil || v.Index == nil {
			return 0, fmt.Errorf("missing episode identity")
		}
		if prior, ok := owners[*v.Seed]; ok && prior != owner {
			return 0, fmt.Errorf("episode seed repeated across batches")
		}
		owners[*v.Seed] = owner
		k := [2]int{*v.Seed, *v.Index}
		if seen[k] {
			return 0, fmt.Errorf("duplicate transition/context")
		}
		seen[k] = true
		n++
	}
	return n, scanner.Err()
}
