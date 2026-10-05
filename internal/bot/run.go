package bot

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"q2coopbot/internal/harness"
	"q2coopbot/internal/harness/checkpoint"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

// Config contains runtime settings for one UDP companion session.
type Config struct {
	CombatMode, CombatProviderFile      string
	CombatCapture                       bool
	Campaign                            bool
	CampaignNextMap                     string
	CampaignRoute                       []string
	CampaignUnitMaps                    []string
	CheckpointControl                   string
	CheckpointRestore                   string
	CheckpointMode                      string
	MemoryFile                          string
	MemorySession                       string
	TestWeaponSwitchFixture             string
	TestDisableProjectileLead           bool
	TestProjectileComparison            bool
	TestCombatBarrier                   bool
	TestSynchronous                     bool
	TestLight                           *int
	TestInvulnerable                    bool
	TestInitialHealth                   int
	TestChangeEntry                     string
	TestSession                         string
	TestSessionRole                     string
	TestScenarioResult                  string
	TestScenarioTailFrames              int
	TestScenario                        string
	TestDisableProbe                    bool
	TestScenarioFrameOrigin             int
	TestSetupHoldFrames                 int
	TestHideHealthFrames                []int
	TestDisableSearch                   bool
	TestWalkTarget                      string
	TestWalkRoute                       bool
	TestWalkRunIn                       bool
	TestWalkThenPlan                    bool
	TestWalkAfterFrames, TestWalkFrames int
	TestRunInSpeed                      int
	Host, Name, GameDir, AASDir         string
	WorldFile, TracePath, StopFile      string
	System1Model, System2Model          string
	TestChangeMap, TestRCONPassword     string
	TestTeleportMap, TestTeleport       string
	TestCampaignGoal                    string
	TestTeleportAfter                   string
	TestTeleportAfterFrames             int
	TestTeleportReturn                  string
	TestTeleportReturnAfterFrames       int
	TestJumpAfterTeleportFrames         int
	TestJumpAgainAfterTeleportFrames    int
	TestSpawnMap, TestSpawnSoldier      string
	TestSpawnClass                      string
	Port, GameFrames, TestChangeAfter   int
	TestGapStart, TestGapFrames         int
	Duration                            time.Duration
	FramePaced, Idle, ExitOnReconnect   bool
	TestLineCross                       bool
	TestHoldPosition                    bool
	TestHoldPositionMap                 string
	TestGroundEdgeProbe                 bool
	TestNoAAS                           bool
	TestDoorProbe                       bool
	TestDoorPassProbe                   bool
	TestDoorPassSpeed                   int
	TestButtonProbe                     bool
	TestButtonAutoGoal                  bool
	TestNoBSP, TestPartialBSP           bool
	TestHideDoor53                      bool
}

func parseTestTeleport(value string) (quake.Vec3, error) {
	var position quake.Vec3
	parts := strings.Split(value, ",")
	if len(parts) != 3 {
		return position, fmt.Errorf("test teleport position must be x,y,z")
	}
	for i, part := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) || v < -32768 || v > 32767 {
			return position, fmt.Errorf("invalid test teleport coordinate %q", part)
		}
		position[i] = v
	}
	return position, nil
}

// The test server accepts the destination and entry as separate RCON arguments.
// Its command joins them after console macro expansion, which would consume '$'.
func transitionMapArgument(destination, previous string) (string, error) {
	validMap := regexp.MustCompile(`^[A-Za-z0-9_]+$`)
	if !validMap.MatchString(destination) || !validMap.MatchString(previous) {
		return "", fmt.Errorf("transition requires simple destination and previous map names")
	}
	return destination + " " + previous, nil
}

// Test-only itinerary: each waypoint is reached by normal gameplay, with no
// extra teleports or scripted button actions between stages.
func parseTestCampaignGoals(value string) ([]quake.Vec3, error) {
	parts := strings.Split(value, ";")
	if len(parts) > 16 {
		return nil, fmt.Errorf("test.campaign_goal allows at most 16 waypoints")
	}
	var points []quake.Vec3
	for _, part := range parts {
		point, err := parseTestTeleport(part)
		if err != nil {
			return nil, err
		}
		points = append(points, point)
	}
	return points, nil
}

func Run(ctx context.Context, cfg Config) error {
	if cfg.CombatMode == "" {
		cfg.CombatMode = "rules"
	}
	if cfg.CombatMode != "rules" && cfg.CombatMode != "learned-shadow" && cfg.CombatMode != "learned" {
		return fmt.Errorf("unknown combat mode")
	}
	var combatProvider policy.Provider
	if cfg.CombatMode != "rules" {
		if cfg.Host != "127.0.0.1" || !cfg.FramePaced || !cfg.CombatCapture || cfg.CombatProviderFile == "" || cfg.TestTeleport == "" || cfg.Idle || cfg.TestScenario != "" || cfg.TestSession != "" || cfg.TestWalkTarget != "" || cfg.TestLineCross || cfg.TestCombatBarrier && !cfg.TestSynchronous || cfg.TestHoldPosition || cfg.TestHoldPositionMap != "" || cfg.TestWeaponSwitchFixture != "" && cfg.TestWeaponSwitchFixture != "parasite_blaster" {
			return fmt.Errorf("direct/shadow combat pilot requires isolated loopback placement, frame pacing, capture, probe and fixed Blaster without scripted command overrides")
		}
		var err error
		combatProvider, err = policy.LoadProvider(cfg.CombatProviderFile, cfg.TestSynchronous)
		if err != nil {
			return fmt.Errorf("combat provider: %w", err)
		}
	} else if cfg.CombatProviderFile != "" {
		return fmt.Errorf("rules mode does not load a combat provider")
	}
	if closer, ok := combatProvider.(interface{ Close() error }); ok {
		defer closer.Close()
	}
	if cfg.CombatCapture && (!cfg.FramePaced || cfg.TracePath == "" || cfg.System1Model != "" || cfg.System2Model != "") {
		return fmt.Errorf("rules combat capture requires frame-paced trace output and both LLM models disabled")
	}
	var testCampaignGoals []quake.Vec3
	if cfg.TestCampaignGoal != "" {
		if !cfg.Campaign || !cfg.FramePaced || cfg.Host != "127.0.0.1" || cfg.TestTeleportMap == "" || cfg.CheckpointRestore != "" || cfg.CheckpointControl != "" {
			return fmt.Errorf("test.campaign_goal requires loopback frame-paced campaign placement without checkpoint control/restore")
		}
		points, err := parseTestCampaignGoals(cfg.TestCampaignGoal)
		if err != nil {
			return err
		}
		testCampaignGoals = points
	}
	var restore *checkpoint.Capture
	if cfg.CheckpointRestore != "" {
		if !cfg.FramePaced || cfg.Host != "127.0.0.1" || cfg.CheckpointControl == "" || (cfg.CheckpointMode != "resume" && cfg.CheckpointMode != "fresh") || cfg.MemoryFile != "" || cfg.TestSession != "" || cfg.TestTeleport != "" || cfg.TestTeleportAfter != "" || cfg.TestSpawnSoldier != "" || cfg.TestInitialHealth != 0 || cfg.TestInvulnerable {
			return fmt.Errorf("checkpoint restore requires explicit resume/fresh, loopback frame pacing and no placement/session/travel-memory overrides")
		}
		capture, err := checkpoint.ReadParticipant(cfg.CheckpointRestore, cfg.Name)
		if err != nil {
			return err
		}
		restore = &capture
	} else if cfg.CheckpointMode != "" {
		return fmt.Errorf("checkpoint_mode requires checkpoint_restore")
	}
	if cfg.CheckpointControl != "" && (!cfg.FramePaced || cfg.Host != "127.0.0.1") {
		return fmt.Errorf("checkpoint control requires frame-paced IPv4 loopback harness")
	}
	if cfg.TestCombatBarrier && (!cfg.FramePaced || cfg.TestTeleport == "" || cfg.TestTeleportMap != "base1") {
		return fmt.Errorf("combat barrier requires frame pacing and base1 teleport")
	}
	if cfg.TestSynchronous && (cfg.Host != "127.0.0.1" || !cfg.CombatCapture || !cfg.FramePaced || !cfg.TestCombatBarrier || cfg.TestWeaponSwitchFixture != "parasite_blaster") {
		return fmt.Errorf("synchronous learning fixture requires loopback, capture, frame pacing, barrier and parasite_blaster")
	}
	if cfg.TestLight != nil && (*cfg.TestLight < 0 || *cfg.TestLight > 255 || !cfg.TestCombatBarrier) {
		return fmt.Errorf("test light requires combat barrier and a value in 0..255")
	}
	if (cfg.TestDisableProjectileLead || cfg.TestProjectileComparison) && (!cfg.FramePaced || !strings.HasPrefix(cfg.TestWeaponSwitchFixture, "projectile_")) {
		return fmt.Errorf("projectile comparison controls require a frame-paced projectile fixture")
	}
	sessionDefinition, sessionRunner, err := configureSession(&cfg)
	if err != nil {
		return err
	}
	if (cfg.TestScenario != "" || sessionRunner != nil) && cfg.TestScenarioTailFrames != 0 {
		return fmt.Errorf("scenario actor publishes completion; tail frames belong to the observing client")
	}
	if cfg.TestScenarioResult != "" && (!cfg.FramePaced || cfg.TestScenario == "" && sessionRunner == nil && (cfg.TestScenarioTailFrames < 2 || cfg.TestScenarioTailFrames > 1000)) {
		return fmt.Errorf("scenario_result requires frame pacing and actor scenario or 2..1000 tail frames")
	}
	if cfg.TestScenarioResult == "" && cfg.TestScenarioTailFrames != 0 {
		return fmt.Errorf("scenario_tail_frames requires scenario_result")
	}
	var scenario *harness.Runner
	if cfg.TestScenario != "" {
		if !cfg.FramePaced || !cfg.Idle || cfg.TestTeleport == "" && restore == nil || cfg.TestTeleportAfter != "" || cfg.TestWalkTarget != "" || cfg.TestLineCross || cfg.TestJumpAfterTeleportFrames != 0 || cfg.TestJumpAgainAfterTeleportFrames != 0 || cfg.TestHoldPosition || cfg.TestGapFrames != 0 || cfg.TestSpawnSoldier != "" {
			return fmt.Errorf("test.scenario requires isolated frame-paced idle actor and initial placement")
		}
		definition, err := harness.Load(cfg.TestScenario)
		if err != nil {
			return fmt.Errorf("scenario: %w", err)
		}
		mapName := cfg.TestTeleportMap
		if restore != nil {
			mapName = restore.Map
		}
		if definition.Map != mapName {
			return fmt.Errorf("scenario map differs from actor placement")
		}
		scenario = harness.New(definition)
	}
	if (cfg.TestDisableSearch || cfg.TestDisableProbe) && !cfg.FramePaced {
		return fmt.Errorf("test.disable_search and test.disable_probe require run.frame_paced")
	}
	walkTarget, err := validateTestWalk(cfg)
	if err != nil {
		return err
	}
	if cfg.TestScenarioFrameOrigin < 0 || cfg.TestScenarioFrameOrigin > 100000 || cfg.TestScenarioFrameOrigin > 0 && (!cfg.FramePaced || cfg.TestTeleport == "") {
		return fmt.Errorf("test.scenario_frame_origin requires frame pacing, initial teleport and 0..100000 frame")
	}
	if cfg.TestInvulnerable && (!cfg.FramePaced || cfg.TestTeleport == "") {
		return fmt.Errorf("test invulnerability requires frame pacing and teleport")
	}
	if cfg.TestWeaponSwitchFixture != "" && (cfg.TestWeaponSwitchFixture != "blaster" && cfg.TestWeaponSwitchFixture != "stocked" && cfg.TestWeaponSwitchFixture != "economy_weak" && cfg.TestWeaponSwitchFixture != "economy_armed" && cfg.TestWeaponSwitchFixture != "parasite_stocked" && cfg.TestWeaponSwitchFixture != "parasite_blaster" && cfg.TestWeaponSwitchFixture != "parasite_hyper" && cfg.TestWeaponSwitchFixture != "parasite_rail" && cfg.TestWeaponSwitchFixture != "parasite_scarce" && cfg.TestWeaponSwitchFixture != "economy_pair_weak" && cfg.TestWeaponSwitchFixture != "economy_pair_heavy" && cfg.TestWeaponSwitchFixture != "projectile_blaster" && cfg.TestWeaponSwitchFixture != "projectile_hyper" && cfg.TestWeaponSwitchFixture != "rail_precision" && cfg.TestWeaponSwitchFixture != "rail_friend_behind" && cfg.TestWeaponSwitchFixture != "hand_grenade_guard" && !handGrenadeArmFixture(cfg.TestWeaponSwitchFixture) && cfg.TestWeaponSwitchFixture != "hand_grenade_observe" && cfg.TestWeaponSwitchFixture != "hand_grenade_auto" || !cfg.FramePaced || cfg.TestTeleport == "") {
		return fmt.Errorf("weapon_switch_fixture requires a supported fixture, frame pacing and teleport")
	}
	if cfg.TestInitialHealth < 0 || cfg.TestInitialHealth > 100 || cfg.TestInitialHealth > 0 && (!cfg.FramePaced || cfg.TestTeleport == "") {
		return fmt.Errorf("initial_health requires frame pacing, teleport, and 1..100 health")
	}
	if cfg.TestChangeEntry != "" {
		if _, err := transitionMapArgument(cfg.TestChangeMap, cfg.TestChangeEntry); err != nil {
			return err
		}
	}
	if cfg.TestSetupHoldFrames < 0 || cfg.TestSetupHoldFrames > 1000 || cfg.TestSetupHoldFrames > 0 && (!cfg.FramePaced || cfg.TestTeleport == "") {
		return fmt.Errorf("test.setup_hold_frames requires frame pacing, initial teleport and 0..1000 frames")
	}
	if len(cfg.TestHideHealthFrames) > 0 && (len(cfg.TestHideHealthFrames) != 2 || !cfg.FramePaced || cfg.TestTeleport == "" || cfg.TestHideHealthFrames[0] < 0 || cfg.TestHideHealthFrames[1] <= cfg.TestHideHealthFrames[0] || cfg.TestHideHealthFrames[1] > 1000) {
		return fmt.Errorf("test.hide_health_frames requires frame pacing, teleport and a bounded [start,end) interval")
	}
	if cfg.GameDir == "" {
		return fmt.Errorf("client.game_dir is required")
	}
	if cfg.GameFrames < 0 || cfg.GameFrames > 0 && !cfg.FramePaced {
		return fmt.Errorf("run.game_frames requires run.frame_paced and a non-negative value")
	}
	if cfg.TracePath != "" && !cfg.FramePaced {
		return fmt.Errorf("output.trace_jsonl requires run.frame_paced")
	}
	if cfg.TestLineCross && (!cfg.FramePaced || !cfg.Idle || cfg.TestSpawnMap != "base1") {
		return fmt.Errorf("test.line_cross requires run.frame_paced, test.idle, and test.spawn_map=base1")
	}
	if cfg.TestHoldPosition && !cfg.FramePaced {
		return fmt.Errorf("test.hold_position requires run.frame_paced")
	}
	if cfg.TestHoldPositionMap != "" && (!cfg.FramePaced || cfg.Host != "127.0.0.1" || cfg.CheckpointControl == "" || validateCampaignRoute([]string{cfg.TestHoldPositionMap, "fixture_end"}) != nil) {
		return fmt.Errorf("test.hold_position_map requires loopback frame-paced checkpoint control and a valid map")
	}
	if cfg.TestNoAAS && !cfg.FramePaced {
		return fmt.Errorf("test.no_aas requires run.frame_paced")
	}
	if (cfg.TestNoBSP || cfg.TestPartialBSP) && (!cfg.FramePaced || cfg.TestNoBSP && cfg.TestPartialBSP) {
		return fmt.Errorf("test.no_bsp and test.partial_bsp require frame pacing and are mutually exclusive")
	}
	if cfg.TestGroundEdgeProbe && (!cfg.FramePaced || cfg.TestTeleportMap != "base1" || cfg.TestTeleport != "-88,40,24") {
		return fmt.Errorf("test.ground_edge_probe requires frame pacing and the base1 edge teleport")
	}
	if cfg.TestDoorProbe && (!cfg.FramePaced || cfg.TestTeleportMap != "base2" || cfg.TestTeleport != "96,-300,24") {
		return fmt.Errorf("test.door_probe requires frame pacing and the base2 door teleport")
	}
	if cfg.TestHideDoor53 && !cfg.TestDoorProbe {
		return fmt.Errorf("test.hide_door_53 requires test.door_probe")
	}
	if cfg.TestDoorPassSpeed != 0 && (!cfg.TestDoorPassProbe || cfg.TestDoorPassSpeed < 80 || cfg.TestDoorPassSpeed > 300) {
		return fmt.Errorf("test.door_pass_speed requires door_pass_probe and 80..300")
	}
	validDoorOrigin := cfg.TestTeleport == "-96,-800,24" || cfg.TestTeleport == "-64,-800,24" || cfg.TestTeleport == "-48,-800,24"
	if cfg.TestDoorPassProbe && (!cfg.FramePaced || cfg.TestTeleportMap != "base2" || !validDoorOrigin) {
		return fmt.Errorf("test.door_pass_probe requires frame pacing and the base2 approach teleport")
	}
	if cfg.TestButtonProbe && (!cfg.FramePaced || cfg.TestTeleportMap != "base2" || cfg.TestTeleport != "320,1940,-144") {
		return fmt.Errorf("test.button_probe requires frame pacing and the base2 button teleport")
	}
	if cfg.TestButtonAutoGoal && (!cfg.FramePaced || cfg.TestTeleportMap != "base2" || cfg.TestTeleport != "194,1940,-144") {
		return fmt.Errorf("test.button_auto_goal requires frame pacing and the base2 door-side teleport")
	}
	if cfg.TestGapStart != 0 || cfg.TestGapFrames != 0 {
		if !cfg.FramePaced || cfg.GameFrames == 0 || cfg.TestGapStart < 1 || cfg.TestGapFrames < 1 || cfg.TestGapStart+cfg.TestGapFrames >= cfg.GameFrames {
			return fmt.Errorf("test observation gap requires run.frame_paced and a positive interval inside run.game_frames")
		}
	}
	if cfg.TestChangeMap != "" {
		validMap := regexp.MustCompile(`^[A-Za-z0-9_]+$`)
		if !cfg.FramePaced || cfg.GameFrames <= cfg.TestChangeAfter || cfg.TestChangeAfter < 1 || !validMap.MatchString(cfg.TestChangeMap) {
			return fmt.Errorf("test.change_map requires run.frame_paced, run.game_frames greater than test.change_after_frames, and a simple map name")
		}
		if cfg.TestRCONPassword == "" {
			return fmt.Errorf("Q2COOPBOT_TEST_RCON is required for test map change")
		}
	}
	var teleportPosition quake.Vec3
	if cfg.TestTeleportMap != "" || cfg.TestTeleport != "" {
		if !cfg.FramePaced || !regexp.MustCompile(`^[A-Za-z0-9_]+$`).MatchString(cfg.TestTeleportMap) {
			return fmt.Errorf("test.teleport requires run.frame_paced and a simple test.teleport_map")
		}
		var err error
		teleportPosition, err = parseTestTeleport(cfg.TestTeleport)
		if err != nil {
			return err
		}
	}
	var teleportAfterPosition quake.Vec3
	if cfg.TestTeleportAfter != "" || cfg.TestTeleportAfterFrames != 0 {
		if cfg.TestTeleport == "" || cfg.TestTeleportAfterFrames < 1 {
			return fmt.Errorf("test.teleport_after requires test.teleport and positive test.teleport_after_frames")
		}
		var err error
		teleportAfterPosition, err = parseTestTeleport(cfg.TestTeleportAfter)
		if err != nil {
			return err
		}
	}
	if cfg.TestJumpAfterTeleportFrames != 0 && (!cfg.FramePaced || !cfg.Idle || cfg.TestTeleportAfter == "" || cfg.TestJumpAfterTeleportFrames < 1 || cfg.TestTeleportReturn != "") {
		return fmt.Errorf("test.jump_after_teleport_frames requires idle client, test.teleport_after and no teleport_return")
	}
	if cfg.TestJumpAgainAfterTeleportFrames != 0 && (cfg.TestJumpAfterTeleportFrames == 0 ||
		cfg.TestJumpAgainAfterTeleportFrames < cfg.TestJumpAfterTeleportFrames+8) {
		return fmt.Errorf("test.jump_again_after_teleport_frames requires a first jump at least eight frames earlier")
	}
	var teleportReturnPosition quake.Vec3
	if cfg.TestTeleportReturn != "" || cfg.TestTeleportReturnAfterFrames != 0 {
		if cfg.TestTeleportAfter == "" || cfg.TestTeleportReturnAfterFrames < 1 {
			return fmt.Errorf("test.teleport_return requires test.teleport_after and positive test.teleport_return_after_frames")
		}
		var err error
		teleportReturnPosition, err = parseTestTeleport(cfg.TestTeleportReturn)
		if err != nil {
			return err
		}
	}
	var spawnPosition quake.Vec3
	if cfg.TestSpawnMap != "" || cfg.TestSpawnSoldier != "" {
		if !cfg.FramePaced || !regexp.MustCompile(`^[A-Za-z0-9_]+$`).MatchString(cfg.TestSpawnMap) {
			return fmt.Errorf("test.spawn_soldier requires run.frame_paced and a simple test.spawn_map")
		}
		var err error
		spawnPosition, err = parseTestTeleport(cfg.TestSpawnSoldier)
		if err != nil {
			return err
		}
	}
	if cfg.TestSpawnClass == "" {
		cfg.TestSpawnClass = "monster_soldier_light"
	}
	if cfg.TestSpawnClass != "monster_soldier_light" && cfg.TestSpawnClass != "monster_soldier_ss" && cfg.TestSpawnClass != "monster_infantry" && cfg.TestSpawnClass != "monster_tank" && cfg.TestSpawnClass != "monster_flyer" && cfg.TestSpawnClass != "monster_parasite" {
		return fmt.Errorf("unsupported test spawn class %q", cfg.TestSpawnClass)
	}
	if cfg.AASDir == "" {
		cfg.AASDir = filepath.Join(cfg.GameDir, "maps")
	}
	address, err := net.ResolveUDPAddr("udp4", fmt.Sprintf("%s:%d", cfg.Host, cfg.Port))
	if err != nil {
		return err
	}
	conn, err := net.ListenUDP("udp4", nil)
	if err != nil {
		return err
	}
	defer conn.Close()
	client := &Client{
		combatControl:      combatControl{mode: cfg.CombatMode, provider: combatProvider},
		combatCapture:      cfg.CombatCapture,
		scenarioResultPath: cfg.TestScenarioResult, scenarioTailFrames: cfg.TestScenarioTailFrames,
		scenario:                scenario,
		sessionDefinition:       sessionDefinition,
		session:                 sessionRunner,
		testScenarioFrameOrigin: cfg.TestScenarioFrameOrigin,
		testSetupHoldFrames:     cfg.TestSetupHoldFrames,
		testHideHealthFrames:    cfg.TestHideHealthFrames,
		testInitialHealth:       cfg.TestInitialHealth,
		testWeaponSwitchFixture: cfg.TestWeaponSwitchFixture,
		testInvulnerable:        cfg.TestInvulnerable,
		testChangeEntry:         cfg.TestChangeEntry,
		testWalkTarget:          walkTarget, testWalkAfterFrames: cfg.TestWalkAfterFrames, testWalkFrames: cfg.TestWalkFrames,
		testRunInSpeed:   cfg.TestRunInSpeed,
		testWalkRoute:    cfg.TestWalkRoute,
		testWalkRunIn:    cfg.TestWalkRunIn,
		testWalkThenPlan: cfg.TestWalkThenPlan,
		conn:             conn, address: address, qport: uint16(rand.Intn(65535) + 1), seq: 1,
		decoder: quake.NewDecoder(), planner: &Planner{Campaign: cfg.Campaign, CampaignNextMap: cfg.CampaignNextMap, CampaignRoute: append([]string(nil), cfg.CampaignRoute...), CampaignUnitMaps: append([]string(nil), cfg.CampaignUnitMaps...), AASDir: cfg.AASDir, GameClock: cfg.FramePaced, TestNoAAS: cfg.TestNoAAS, TestNoBSP: cfg.TestNoBSP, TestPartialBSP: cfg.TestPartialBSP, TestHideDoor53: cfg.TestHideDoor53, TestDisableProjectileLead: cfg.TestDisableProjectileLead, TestDisableHandGrenade: cfg.Idle || cfg.TestWeaponSwitchFixture == "hand_grenade_observe" || cfg.TestWeaponSwitchFixture == "hand_grenade_guard" || handGrenadeArmFixture(cfg.TestWeaponSwitchFixture)},
		root: cfg.GameDir, worldFile: cfg.WorldFile, stopFile: cfg.StopFile, name: cfg.Name, checkpointControl: cfg.CheckpointControl, checkpointRestore: restore, checkpointMode: cfg.CheckpointMode,
		memoryFile: cfg.MemoryFile, memorySession: cfg.MemorySession,
		idle: cfg.Idle, duration: cfg.Duration, framePaced: cfg.FramePaced, gameFrames: cfg.GameFrames,
		exitOnReconnect: cfg.ExitOnReconnect, testChangeMap: cfg.TestChangeMap,
		testChangeAfter: cfg.TestChangeAfter, testRconPassword: cfg.TestRCONPassword,
		testTeleportMap: cfg.TestTeleportMap, testTeleportPosition: teleportPosition,
		testTeleportAfterPosition: teleportAfterPosition, testTeleportAfterFrames: cfg.TestTeleportAfterFrames,
		testTeleportReturnPosition: teleportReturnPosition, testTeleportReturnAfterFrames: cfg.TestTeleportReturnAfterFrames,
		testJumpAfterTeleportFrames:      cfg.TestJumpAfterTeleportFrames,
		testJumpAgainAfterTeleportFrames: cfg.TestJumpAgainAfterTeleportFrames,
		testSpawnMap:                     cfg.TestSpawnMap, testSpawnPosition: spawnPosition,
		testSpawnClass: cfg.TestSpawnClass,
		testGapStart:   cfg.TestGapStart, testGapFrames: cfg.TestGapFrames,
		testLineCross:            cfg.TestLineCross,
		testHoldPosition:         cfg.TestHoldPosition,
		testHoldPositionMap:      cfg.TestHoldPositionMap,
		testProjectileComparison: cfg.TestProjectileComparison,
		testCombatBarrier:        cfg.TestCombatBarrier,
		testSynchronous:          cfg.TestSynchronous,
		testLight:                cfg.TestLight,
		testGroundEdgeProbe:      cfg.TestGroundEdgeProbe,
		testDoorProbe:            cfg.TestDoorProbe,
		testDoorPassProbe:        cfg.TestDoorPassProbe,
		testButtonProbe:          cfg.TestButtonProbe,
		testButtonAutoGoal:       cfg.TestButtonAutoGoal,
	}
	client.planner.TestDisableSearch = cfg.TestDisableSearch
	client.planner.testDoorPassSpeed = float64(cfg.TestDoorPassSpeed)
	client.planner.TestDisableProbe = cfg.TestDisableProbe
	client.planner.testCampaignGoals = testCampaignGoals
	if cfg.TracePath != "" {
		client.traceFile, err = os.Create(cfg.TracePath)
		if err != nil {
			return err
		}
		defer client.traceFile.Close()
	}
	if cfg.System2Model != "" {
		client.strategist = NewStrategist(cfg.System2Model)
	}
	if cfg.System1Model != "" {
		client.tactician = NewTactician(cfg.System1Model)
	}
	if err := client.run(ctx); err != nil {
		return err
	}
	if cfg.WorldFile != "" {
		_ = client.writeWorld()
	}
	gameFPS := 0.0
	if client.framePaced && !client.firstMoveAt.IsZero() && client.lastMoveFrame > client.firstMoveFrame {
		gameFPS = float64(client.lastMoveFrame-client.firstMoveFrame) / time.Since(client.firstMoveAt).Seconds()
	}
	log.Printf("finished connected=%t begun=%t frames=%d moves=%d attacks=%d frame_paced=%t game_frames=%d frame_gaps=%d server_suppressed=%d game_fps=%.2f wall_s=%.2f map_changes=%d last_map=%s transition_timeout=%t decode_errors=%d last_error=%q", client.connected, client.begun, client.frames, client.moves, client.attacks, client.framePaced, max(0, client.lastMoveFrame-client.firstMoveFrame), client.frameGaps, client.suppressedFrames, gameFPS, time.Since(client.start).Seconds(), client.mapChanges, client.lastObservedMap, client.testChangeTimedOut, client.decoder.Errors, client.decoder.LastError)
	return nil
}
