package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"q2coopbot/internal/trainingepisodes"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() error {
	registry := flag.String("registry", "scripts/scenarios/combat-training/index.json", "training episode registry")
	list := flag.Bool("list", false, "validate and list available/planned episodes")
	verify := flag.String("verify-plan", "", "recompute and verify a frozen generated execution plan")
	ids := flag.String("episodes", "", "comma-separated episode IDs")
	split := flag.String("split", "validation", "train, validation, test or confirmation")
	mode := flag.String("mode", "both", "rules, learned or both")
	model := flag.String("model", "", "frozen PPO weights for learned")
	count := flag.Int("count", 4, "episodes per descriptor, 1..80 (cohort scheduler requires multiples of four)")
	recordCombatAdvice := flag.Bool("record-combat-advice", false, "record hypothetical reactive expert recommendations on learner states; never applied")
	checkedWallSlide := flag.Bool("checked-wall-slide", false, "experimental checked wall-slide combat guard")
	executionContract := flag.String("execution-contract", "", "model-matched combat execution contract; historical default legacy_v1")
	diagnosticNominalAim := flag.Bool("diagnostic-nominal-aim", false, "evaluation-only observed-target aiming intervention, not a learned controller")
	diagnosticStoppedCrouch := flag.Bool("diagnostic-stopped-crouch", false, "evaluation-only preservation of checked model-selected stopped crouch throughout first life")
	diagnosticCheckedGroundRequest := flag.Bool("diagnostic-checked-ground-request", false, "evaluation-only checked ground request with stopped-crouch fallback throughout first life")
	offset := flag.Int("seed-offset", 0, "offset inside the registered split")
	root := flag.String("root", ".", "repository root")
	out := flag.String("out", "", "fresh execution plan JSON")
	artifacts := flag.String("artifacts", "", "fresh runtime output root")
	campaign := flag.Int("campaign-sites", 0, "generate this many distinct original encounter sites per selected map into out")
	allMapSites := flag.Int("all-map-sites", 0, "scan all 39 campaign maps only; generate up to this many checked ground sites per map")
	nativeRoster := flag.Bool("native-random-roster", false, "with all-map-sites, generate map-native repeated groups of2..4 and partial inventory")
	nativeProfiles := flag.Bool("native-monster-profiles", false, "with all-map-sites, scan each native combat type using its own stock hull/habitat")
	habitatRoster := flag.Bool("native-habitat-roster", false, "with all-map-sites, add random native profile groups using per-class habitat placements")
	campaignMaps := flag.String("campaign-maps", "base1,base2", "comma-separated stock campaign maps for site generation")
	campaignGroup := flag.String("campaign-group", "", "comma-separated distinct ground monster classes; generate group recipes with Super Shotgun inventory")
	campaignGroupLoadout := flag.String("campaign-group-loadout", "weapons-ssg", "group starting inventory: blaster, machinegun, weapons, weapons-ssg or weapons-grenades-recovery")
	flag.Parse()
	if *habitatRoster && (*allMapSites == 0 || *nativeProfiles || *nativeRoster) {
		return fmt.Errorf("native-habitat-roster requires all-map-sites and excludes other native catalog modes")
	}
	if *nativeProfiles && (*allMapSites == 0 || *nativeRoster) {
		return fmt.Errorf("native-monster-profiles requires all-map-sites and excludes native-random-roster")
	}
	if *nativeRoster && *allMapSites == 0 {
		return fmt.Errorf("native-random-roster requires all-map-sites")
	}
	if *allMapSites != 0 {
		if *campaign != 0 || *out == "" {
			return fmt.Errorf("all-map-sites needs out and cannot combine with campaign-sites")
		}
		abs, err := filepath.Abs(*root)
		if err != nil {
			return err
		}
		var episodes []trainingepisodes.Episode
		var coverage []trainingepisodes.MapCoverage
		if *habitatRoster {
			episodes, coverage, err = trainingepisodes.NativeHabitatRosterRecipes(abs, *allMapSites)
		} else if *nativeProfiles {
			episodes, coverage, err = trainingepisodes.NativeProfileRecipes(abs, *allMapSites)
		} else if *nativeRoster {
			episodes, coverage, err = trainingepisodes.NativeRosterRecipes(abs, *allMapSites, 2, 4)
		} else {
			episodes, coverage, err = trainingepisodes.AllMapRecipes(abs, *allMapSites)
		}
		if err != nil {
			return err
		}
		file, err := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		defer file.Close()
		encoder := json.NewEncoder(file)
		encoder.SetIndent("", "  ")
		return encoder.Encode(struct {
			Version  int                            `json:"version"`
			Coverage []trainingepisodes.MapCoverage `json:"coverage"`
			Episodes []trainingepisodes.Episode     `json:"episodes"`
		}{1, coverage, episodes})
	}
	if *campaign != 0 {
		abs, err := filepath.Abs(*root)
		if err != nil {
			return err
		}
		maps := strings.Split(*campaignMaps, ",")
		for i := range maps {
			maps[i] = strings.TrimSpace(maps[i])
		}
		episodes, err := trainingepisodes.CampaignRecipesForMaps(abs, *campaign, maps)
		if *campaignGroup != "" {
			episodes, err = trainingepisodes.CampaignGroupRecipesForLoadout(abs, *campaign, maps, strings.Split(*campaignGroup, ","), *campaignGroupLoadout)
		}
		if err != nil {
			return err
		}
		f, err := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		defer f.Close()
		e := json.NewEncoder(f)
		e.SetIndent("", "  ")
		return e.Encode(struct {
			Version  int                        `json:"version"`
			Episodes []trainingepisodes.Episode `json:"episodes"`
		}{1, episodes})
	}
	if *verify != "" {
		abs, err := filepath.Abs(*root)
		if err != nil {
			return err
		}
		return trainingepisodes.VerifyPlan(*verify, abs)
	}
	r, err := trainingepisodes.Load(*registry)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if *list {
		return enc.Encode(r)
	}
	if *out == "" || *artifacts == "" {
		return fmt.Errorf("out and artifacts required")
	}
	abs, err := filepath.Abs(*root)
	if err != nil {
		return err
	}
	p, err := trainingepisodes.Build(r, abs, strings.Split(*ids, ","), *split, *mode, *model, *artifacts, *count, *offset)
	if err != nil {
		return err
	}
	p.RecordCombatAdvice = *recordCombatAdvice
	if p.RecordCombatAdvice && *mode != "learned" {
		return fmt.Errorf("combat advice requires learned plans")
	}
	p.CheckedWallSlide = *checkedWallSlide
	p.ExecutionContract = *executionContract
	p.DiagnosticStoppedCrouch = *diagnosticStoppedCrouch
	p.DiagnosticCheckedGroundRequest = *diagnosticCheckedGroundRequest
	if p.DiagnosticCheckedGroundRequest && (p.DiagnosticStoppedCrouch || *split == "train" || *mode != "learned") {
		return fmt.Errorf("diagnostic checked ground request requires exclusive learned evaluation plan")
	}
	if p.DiagnosticStoppedCrouch && (*split == "train" || *mode != "learned") {
		return fmt.Errorf("diagnostic stopped crouch requires learned evaluation split")
	}
	p.DiagnosticNominalAim = *diagnosticNominalAim
	if p.DiagnosticNominalAim && (*split == "train" || *mode != "learned") {
		return fmt.Errorf("diagnostic nominal aim requires learned evaluation split")
	}
	if err := trainingepisodes.ValidatePlanExecutionContract(p); err != nil {
		return err
	}
	f, err := os.OpenFile(*out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	e := json.NewEncoder(f)
	e.SetIndent("", "  ")
	return e.Encode(p)
}
