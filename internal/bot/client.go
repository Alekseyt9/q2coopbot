package bot

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"q2coopbot/internal/harness"
	"q2coopbot/internal/quake"
)

type Client struct {
	weaponSwitch                     weaponSwitch
	inventoryWatch                   inventoryWatch
	testWeaponSwitchFixture          string
	sessionConnection                int
	connection                       int
	sessionObserverCycle             observerRespawnState
	sessionTransitionRequested       bool
	sessionTransitionAcked           bool
	sessionReadySent                 bool
	sessionStartFrame                int
	sessionSetupAt                   time.Time
	sessionDefinition                *harness.Session
	session                          *harness.SessionRunner
	sessionPhase                     int
	sessionMap                       string
	sessionGeneration                int
	sessionPendingMap                string
	scenarioResultPath               string
	scenarioTailFrames               int
	scenarioResultSent               bool
	scenarioCompletion               *scenarioCompletion
	scenario                         *harness.Runner
	scenarioPath                     testWalkPath
	conn                             *net.UDPConn
	address                          *net.UDPAddr
	seq, serverSeq                   uint32
	serverReliable                   uint32
	qport                            uint16
	challenge                        int
	connected, begun                 bool
	spawncount                       int
	lastHandshake                    string
	handshakeAt                      time.Time
	decoder                          *quake.Decoder
	pendingSounds                    []quake.SoundEvent
	planner                          *Planner
	root                             string
	previous                         quake.UserCmd
	nextMove                         time.Time
	lastFrame                        time.Time
	framePaced                       bool
	gameFrames                       int
	frameReady                       bool
	latestFrame                      int
	firstMoveFrame                   int
	lastMoveFrame                    int
	firstMoveAt                      time.Time
	frameGaps                        int
	suppressedFrames                 int
	frames, moves                    int
	attacks                          int
	worldFile                        string
	traceFile                        *os.File
	stopFile                         string
	name                             string
	idle                             bool
	testChangeMap                    string
	testChangeEntry                  string
	testInitialHealth                int
	testInvulnerable                 bool
	testChangeAfter                  int
	testRconPassword                 string
	testChangeSent                   bool
	testChangeAt                     time.Time
	testChangeTimedOut               bool
	testTeleportMap                  string
	testTeleportPosition             quake.Vec3
	testTeleportSent                 bool
	testTeleportSentFrame            int
	testTeleportAfterPosition        quake.Vec3
	testTeleportAfterFrames          int
	testTeleportAfterSent            bool
	testTeleportAfterSentFrame       int
	testTeleportReturnPosition       quake.Vec3
	testTeleportReturnAfterFrames    int
	testTeleportReturnSent           bool
	testJumpAfterTeleportFrames      int
	testJumpAgainAfterTeleportFrames int
	testSpawnMap                     string
	testSpawnClass                   string
	testSpawnPosition                quake.Vec3
	testSpawnSent                    bool
	testGapStart                     int
	testGapFrames                    int
	testLineCross                    bool
	testHoldPosition                 bool
	testWalkTarget                   quake.Vec3
	testWalkRoute                    bool
	testWalkPath                     testWalkPath
	testWalkAfterFrames              int
	testWalkFrames                   int
	testSetupHoldFrames              int
	testScenarioFrameOrigin          int
	testGroundEdgeProbe              bool
	testDoorProbe                    bool
	testDoorPassProbe                bool
	testButtonProbe                  bool
	testButtonAutoGoal               bool
	testDoorPassStarted              bool
	testLineEntered                  bool
	testLineLeft                     bool
	lastObservedMap                  string
	mapChanges                       int
	beginPending                     string
	beginAt                          time.Time
	reconnected                      bool
	exitOnReconnect                  bool
	stopOnReconnect                  bool
	strategist                       *Strategist
	tactician                        *Tactician
	seenCommands                     map[string]bool
	duration                         time.Duration
	start                            time.Time
}

func (c *Client) sendRaw(data []byte) error { _, e := c.conn.WriteToUDP(data, c.address); return e }
func (c *Client) oob(message string) error {
	return c.sendRaw(append([]byte{255, 255, 255, 255}, []byte(message)...))
}
func (c *Client) send(payload []byte, reliable bool) error {
	header := make([]byte, 0, 10+len(payload))
	first := c.seq
	if reliable {
		first |= 0x80000000
	}
	header = binary.LittleEndian.AppendUint32(header, first)
	header = binary.LittleEndian.AppendUint32(header, c.serverSeq|(c.serverReliable<<31))
	header = binary.LittleEndian.AppendUint16(header, c.qport)
	c.seq++
	return c.sendRaw(append(header, payload...))
}
func (c *Client) command(command string) error {
	log.Printf("client command: %s", command)
	c.lastHandshake = command
	c.handshakeAt = time.Now()
	return c.send(append(append([]byte{4}, []byte(command)...), 0), true)
}
func (c *Client) reconnect() error {
	_ = c.command("disconnect")
	newConn, err := net.ListenUDP("udp4", nil)
	if err != nil {
		return err
	}
	oldConn := c.conn
	c.conn = newConn
	c.qport = uint16(rand.Intn(65535) + 1)
	oldConn.Close()
	c.connected = false
	c.reconnected = true
	c.begun = false
	c.frameReady = false
	c.beginPending = ""
	c.firstMoveFrame = -1
	c.lastMoveFrame = -1
	c.seq = 1
	c.serverSeq = 0
	c.serverReliable = 0
	c.previous = quake.UserCmd{}
	c.decoder = quake.NewDecoder()
	c.pendingSounds = nil
	c.seenCommands = map[string]bool{}
	c.lastHandshake = ""
	c.handshakeAt = time.Now()
	c.planner.setMap("", c.root)
	log.Printf("full reconnect started")
	return c.oob("getchallenge\n")
}
func (c *Client) handle(packet []byte) {
	if len(packet) >= 4 && string(packet[:4]) == "\xff\xff\xff\xff" {
		message := string(packet[4:])
		log.Printf("server OOB: %q", strings.TrimSpace(message))
		if strings.Contains(message, "challenge ") && !c.connected {
			re := regexp.MustCompile(`challenge\s+(-?\d+)`)
			m := re.FindStringSubmatch(message)
			if len(m) > 1 {
				c.challenge, _ = strconv.Atoi(m[1])
				_ = c.oob(quake.ConnectRequest(c.qport, c.challenge, c.name))
			}
		}
		if strings.Contains(message, "client_connect") && !c.connected {
			c.connection++
			c.connected = true
			_ = c.command("new")
		}
		return
	}
	if len(packet) < 8 {
		return
	}
	first := binary.LittleEndian.Uint32(packet)
	sequence := first & 0x7fffffff
	if sequence <= c.serverSeq {
		return
	}
	c.serverSeq = sequence
	if first&0x80000000 != 0 {
		c.serverReliable ^= 1
	}
	payload := packet[8:]
	frames, e := c.decoder.Parse(payload)
	if e != nil {
		c.decoder.Errors++
		c.decoder.LastError = e.Error()
	}
	if c.decoder.ServerdataSeen {
		c.pendingSounds = nil
		c.spawncount = c.decoder.Spawncount
		c.begun = false
		c.frameReady = false
		c.previous = quake.UserCmd{}
		c.firstMoveFrame = -1
		c.lastMoveFrame = -1
		c.planner.setMap("", c.root)
		c.weaponSwitch = weaponSwitch{}
		c.inventoryWatch = inventoryWatch{}
		c.seenCommands = map[string]bool{}
		command := fmt.Sprintf("configstrings %d 0", c.spawncount)
		c.seenCommands["cmd "+command] = true
		_ = c.command(command)
	}
	c.pendingSounds = append(c.pendingSounds, c.decoder.Sounds...)
	for _, f := range frames {
		c.frames++
		c.suppressedFrames += int(f.Suppressed)
		c.lastFrame = time.Now()
		if c.begun {
			c.latestFrame = f.Number
			c.frameReady = true
		}
		s := c.decoder.Snapshot(f)
		if len(c.pendingSounds) > 0 {
			s.Sounds = c.pendingSounds
			c.pendingSounds = nil
		}
		if c.testButtonAutoGoal && c.testTeleportSent && s.Map == "base2" {
			goal := quake.Vec3{194, 2080, -168}
			s.Teammate = &goal
			s.LastTeammate = &goal
			age := 0
			s.TeammateAgeFrames = &age
		}
		if s.Map != "" {
			if s.Map != c.lastObservedMap {
				if c.lastObservedMap != "" {
					c.mapChanges++
				}
				log.Printf("map observed=%s frame=%d transitions=%d", s.Map, f.Number, c.mapChanges)
				c.lastObservedMap = s.Map
			}
			relativeFrame := f.Number - c.firstMoveFrame
			if c.firstMoveFrame < 0 || c.testGapFrames == 0 || relativeFrame < c.testGapStart || relativeFrame >= c.testGapStart+c.testGapFrames {
				c.planner.testSetupHold = c.testSetupHoldFrames > 0 && s.Map == c.testTeleportMap &&
					(!c.testTeleportSent || c.testScenarioAge(s.Frame) < c.testSetupHoldFrames)
				c.planner.update(s, c.root)
			}
			if c.frames%50 == 0 {
				w := c.planner.World
				log.Printf("frame=%d self=%v teammate=%v goal=%s nav=%s route=%d enemies=%d", f.Number, s.Self, s.Teammate, w.Goal, w.Navigation, len(w.Route), len(s.Enemies))
			}
			if c.worldFile != "" && c.frames%10 == 0 {
				_ = c.writeWorld()
			}
		}
	}
	for _, request := range c.decoder.Commands {
		if c.seenCommands[request] {
			continue
		}
		c.seenCommands[request] = true
		switch {
		case request == "changing":
			c.begun = false
			c.beginPending = ""
			c.frameReady = false
			c.firstMoveFrame = -1
			c.lastMoveFrame = -1
			c.seenCommands = map[string]bool{}
			c.planner.setMap("", c.root)
		case request == "reconnect":
			if c.exitOnReconnect {
				log.Printf("test client leaving on map reconnect")
				c.stopOnReconnect = true
				return
			}
			_ = c.reconnect()
			return
		case strings.HasPrefix(request, "cmd configstrings "), strings.HasPrefix(request, "cmd baselines "):
			_ = c.command(strings.TrimPrefix(request, "cmd "))
		case strings.HasPrefix(request, "precache "):
			parts := strings.Fields(request)
			if len(parts) > 1 {
				c.frameReady = false
				c.nextMove = time.Now()
				if c.reconnected {
					c.beginPending = "begin " + parts[1]
					c.beginAt = time.Now().Add(1500 * time.Millisecond)
					log.Printf("scenario delaying begin after map change")
				} else {
					c.begun = true
					_ = c.command("begin " + parts[1])
				}
			}
		}
	}
}
func (c *Client) writeWorld() error {
	data, e := json.MarshalIndent(c.planner.World, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(c.worldFile, data, 0644)
}

func (c *Client) needsSafetyStop(now time.Time) bool {
	if !c.begun || !c.framePaced || !c.frameReady || c.latestFrame != c.lastMoveFrame ||
		c.planner.World.Updated.IsZero() || now.Sub(c.planner.World.Updated) <= 300*time.Millisecond {
		return false
	}
	return c.previous.Buttons != 0 || c.previous.Forward != 0 || c.previous.Side != 0 || c.previous.Up != 0
}

func (c *Client) run(ctx context.Context) error {
	defer func() { _ = c.conn.Close() }()
	c.start = time.Now()
	c.nextMove = c.start
	c.firstMoveFrame = -1
	c.lastMoveFrame = -1
	c.handshakeAt = c.start
	_ = c.oob("getchallenge\n")
	buffer := make([]byte, 65535)
	for {
		if ctx.Err() != nil || c.stopOnReconnect || c.duration > 0 && time.Since(c.start) >= c.duration || c.stopFile != "" && exists(c.stopFile) {
			if c.connected {
				_ = c.command("disconnect")
			}
			return nil
		}
		_ = c.conn.SetReadDeadline(time.Now().Add(20 * time.Millisecond))
		n, _, e := c.conn.ReadFromUDP(buffer)
		if e == nil {
			c.handle(buffer[:n])
		} else if ne, ok := e.(net.Error); !ok || !ne.Timeout() {
			return e
		}
		now := time.Now()
		if c.begun && c.strategist != nil {
			if decision, ok := c.strategist.poll(c.planner.World); ok {
				c.planner.applyDecision(decision)
			}
			c.strategist.tick(c.planner.World)
		}
		if c.begun && c.tactician != nil {
			if decision, ok := c.tactician.poll(c.planner.World); ok {
				c.planner.applyTactic(decision)
			}
			c.tactician.tick(c.planner.World)
		}
		if c.connected && !c.begun && c.lastHandshake != "" && now.Sub(c.handshakeAt) >= 2*time.Second {
			_ = c.command(c.lastHandshake)
		}
		if c.beginPending != "" && !now.Before(c.beginAt) {
			c.begun = true
			_ = c.command(c.beginPending)
			c.beginPending = ""
		}
		if c.testChangeSent && c.mapChanges == 0 && now.Sub(c.testChangeAt) > 10*time.Second {
			c.testChangeTimedOut = true
			log.Printf("scenario map transition timed out from=%s to=%s", c.lastObservedMap, c.testChangeMap)
			return nil
		}
		if err := c.prepareSessionPhase(); err != nil {
			return err
		}
		if err := c.sessionBarrier(now); err != nil {
			return err
		}
		if c.session != nil {
			c.session.Poll(now.Sub(c.start))
			if c.session.Status.State == "failed" {
				return fmt.Errorf("session: %s", c.session.Status.Reason)
			}
		}
		pause, ready, err := c.sessionTransitionBarrier()
		if err != nil {
			return err
		}
		if ready {
			if c.sessionDefinition.Phases[c.sessionPhase+1].Entry == "reconnect" {
				if err := c.releaseSessionReconnect(); err != nil {
					return err
				}
				if err := c.reconnect(); err != nil {
					return err
				}
				c.sessionPendingMap = ""
				continue
			}
			mapArg, err := transitionMapArgument(c.sessionPendingMap, c.planner.World.Map)
			if err != nil {
				return err
			}
			if err := c.oob(fmt.Sprintf("rcon %s sv_test_map_entry %s\n", c.testRconPassword, mapArg)); err != nil {
				return err
			}
			c.sessionPendingMap = ""
			continue
		}
		if pause {
			continue
		}
		if c.begun && c.testTeleportMap != "" && !c.testTeleportSent && c.planner.World.Map == c.testTeleportMap && c.frameReady {
			p := c.testTeleportPosition
			if err := c.command(fmt.Sprintf("teleport %g %g %g", p[0], p[1], p[2])); err != nil {
				return err
			}
			c.testTeleportSent = true
			if c.testInvulnerable {
				if err := c.command("god"); err != nil {
					return err
				}
			}
			if c.testInitialHealth > 0 {
				if err := c.command(fmt.Sprintf("give health %d", c.testInitialHealth)); err != nil {
					return err
				}
			}
			if c.testWeaponSwitchFixture != "" {
				setup := []string{"give Shotgun", "give Shells 1"}
				if c.testWeaponSwitchFixture == "stocked" {
					setup = append(setup, "give Machinegun", "give Bullets 20")
				}
				setup = append(setup, "use Shotgun")
				var payload []byte
				for _, command := range setup {
					payload = append(payload, 4)
					payload = append(payload, command...)
					payload = append(payload, 0)
				}
				if err := c.send(payload, true); err != nil {
					return err
				}
			}
			c.testTeleportSentFrame = c.latestFrame
			log.Printf("scenario test teleport map=%s target=%v", c.testTeleportMap, p)
			continue
		}
		if c.begun && c.testTeleportSent && !c.testTeleportAfterSent && c.testTeleportAfterFrames > 0 &&
			c.planner.World.Map == c.testTeleportMap && c.frameReady && c.testScenarioAge(c.latestFrame) >= c.testTeleportAfterFrames {
			p := c.testTeleportAfterPosition
			if err := c.command(fmt.Sprintf("teleport %g %g %g", p[0], p[1], p[2])); err != nil {
				return err
			}
			c.testTeleportAfterSent = true
			c.testTeleportAfterSentFrame = c.latestFrame
			log.Printf("scenario second test teleport map=%s target=%v", c.testTeleportMap, p)
			// Shared-clock trials must still send and record this frame's usercmd.
			// Waiting for another packet can otherwise skip the teleport frame.
			if c.testScenarioFrameOrigin == 0 {
				continue
			}
		}
		if c.begun && c.testTeleportAfterSent && !c.testTeleportReturnSent && c.testTeleportReturnAfterFrames > 0 &&
			c.planner.World.Map == c.testTeleportMap && c.frameReady && c.latestFrame-c.testTeleportAfterSentFrame >= c.testTeleportReturnAfterFrames {
			p := c.testTeleportReturnPosition
			if err := c.command(fmt.Sprintf("teleport %g %g %g", p[0], p[1], p[2])); err != nil {
				return err
			}
			c.testTeleportReturnSent = true
			log.Printf("scenario return test teleport map=%s target=%v", c.testTeleportMap, p)
			continue
		}
		if c.begun && c.testSpawnMap != "" && !c.testSpawnSent && c.planner.World.Map == c.testSpawnMap && c.frameReady {
			p := c.testSpawnPosition
			if err := c.command(fmt.Sprintf("spawnentity %s %g %g %g", c.testSpawnClass, p[0], p[1], p[2])); err != nil {
				return err
			}
			c.testSpawnSent = true
			log.Printf("scenario test soldier spawn map=%s target=%v", c.testSpawnMap, p)
			continue
		}
		if c.begun && c.testLineCross && c.firstMoveFrame >= 0 && c.frameReady {
			relativeFrame := c.latestFrame - c.firstMoveFrame
			if relativeFrame >= 3 && !c.testLineEntered {
				if err := c.command("teleport 66 -234 24"); err != nil {
					return err
				}
				c.testLineEntered = true
			} else if relativeFrame >= 5 && !c.testLineLeft {
				if err := c.command("teleport 128 -320 24"); err != nil {
					return err
				}
				c.testLineLeft = true
			}
		}
		if c.begun && c.framePaced && c.testChangeMap != "" && !c.testChangeSent &&
			c.firstMoveFrame >= 0 && c.lastMoveFrame-c.firstMoveFrame >= c.testChangeAfter &&
			c.latestFrame > c.lastMoveFrame {
			entry := c.lastObservedMap
			if c.testChangeEntry != "" {
				entry = c.testChangeEntry
			}
			mapArg, err := transitionMapArgument(c.testChangeMap, entry)
			if err != nil {
				return err
			}
			if err := c.oob(fmt.Sprintf("rcon %s sv_test_map_entry %s\n", c.testRconPassword, mapArg)); err != nil {
				return err
			}
			c.testChangeSent = true
			c.testChangeAt = now
			log.Printf("scenario map change requested from=%s to=%s entry=%s after=%d frames", c.lastObservedMap, c.testChangeMap, c.lastObservedMap, c.testChangeAfter)
			continue
		}
		if c.framePaced && c.gameFrames > 0 && c.firstMoveFrame >= 0 &&
			c.lastMoveFrame-c.firstMoveFrame >= c.gameFrames && c.latestFrame > c.lastMoveFrame {
			if c.connected {
				_ = c.command("disconnect")
			}
			return nil
		}
		if c.framePaced && c.frameReady && c.scenarioResultPath != "" && c.scenario == nil && c.session == nil {
			stop, err := c.scenarioShouldStop()
			if err != nil {
				return err
			}
			if stop {
				log.Printf("scenario_stop_frame=%d scenario_terminal_frame=%d", c.lastMoveFrame, c.scenarioCompletion.EndFrame)
				if c.connected {
					_ = c.command("disconnect")
				}
				return nil
			}
		}
		if c.begun && (!c.framePaced && now.After(c.nextMove) || c.framePaced && c.frameReady && c.latestFrame > c.lastMoveFrame || c.needsSafetyStop(now)) {
			frame := c.latestFrame
			safetyStop := c.needsSafetyStop(now)
			if c.testGroundEdgeProbe && c.testTeleportSent {
				c.planner.setTestGroundEdgeGoal()
			}
			if c.testDoorProbe && c.testTeleportSent {
				c.planner.setTestDoorGoal()
			}
			if c.testDoorPassProbe && c.testTeleportSent {
				s := c.planner.World.Snapshot
				if s.Map == "base2" && s.OnGround && math.Abs(s.Self[0]+64) < 8 && math.Abs(s.Self[1]+800) < 8 {
					c.testDoorPassStarted = true
				}
				if c.testDoorPassStarted {
					c.planner.setTestDoorPassGoal()
				}
			}
			if c.testButtonProbe && c.testTeleportSent {
				c.planner.setTestButtonGoal()
			}
			cmd := c.planner.command(c.previous)
			weaponRequest := ""
			if !safetyStop && !c.idle && !c.planner.testSetupHold && now.Sub(c.planner.World.Updated) <= 300*time.Millisecond {
				weaponRequest = c.weaponSwitch.command(c.planner.World.Snapshot)
				if request := c.inventoryWatch.command(c.planner.World.Snapshot); request != "" {
					if err := c.command(request); err != nil {
						return err
					}
				}
				if weaponRequest != "" {
					if err := c.command(weaponRequest); err != nil {
						return err
					}
				}
			}
			if (c.testGroundEdgeProbe || c.testDoorProbe || c.testDoorPassProbe || c.testButtonProbe) && c.testTeleportSent && !c.planner.World.Snapshot.OnGround {
				cmd = quake.UserCmd{}
				c.planner.World.Command = CommandDecision{MoveSource: "none", AimSource: "none", LimitReason: "test_teleport_settling"}
			}
			if c.testHoldPosition {
				cmd.Forward, cmd.Side, cmd.Up = 0, 0, 0
				c.planner.World.Command.MoveSource = "test_hold"
			}
			if c.planner.testSetupHold {
				cmd = quake.UserCmd{}
				c.planner.World.Command = CommandDecision{MoveSource: "none", AimSource: "none", LimitReason: "test_setup_hold"}
			}
			if c.idle {
				cmd = quake.UserCmd{}
				c.planner.World.Command = CommandDecision{MoveSource: "none", AimSource: "none", LimitReason: "test_idle"}
			}
			var observerKill, observerRespawn bool
			if !safetyStop && c.sessionDefinition != nil && c.session == nil {
				fault := c.sessionDefinition.Phases[c.sessionPhase].ObserverRespawn
				var err error
				observerKill, observerRespawn, err = c.sessionObserverCycle.tick(fault, c.sessionStartFrame, frame, c.planner.World.Snapshot.Health)
				if err != nil {
					return err
				}
				if observerKill {
					if err := c.command("kill"); err != nil {
						return err
					}
				}
				if observerRespawn {
					cmd.Buttons = 1
					c.planner.World.Command.LimitReason = "test_observer_respawn"
				}
			}
			if c.sessionDefinition != nil && c.sessionDefinition.ReadinessBarrier && (c.sessionStartFrame == 0 || frame < c.sessionStartFrame) {
				cmd = quake.UserCmd{}
				c.planner.World.Command = CommandDecision{MoveSource: "none", AimSource: "none", LimitReason: "session_barrier"}
			}
			jumpAge := c.latestFrame - c.testTeleportAfterSentFrame
			if (c.scenario != nil || c.session != nil) && !safetyStop {
				s := c.planner.World.Snapshot
				in := harness.Input{Frame: frame, Generation: c.spawncount, Map: s.Map, Self: s.Self, OnGround: s.OnGround, Health: s.Health, PhaseStart: c.sessionStartFrame, Connection: c.connection}
				var d harness.Decision
				var status *harness.Status
				if c.session != nil {
					decision := c.session.Tick(in, now.Sub(c.start))
					d, c.sessionPendingMap = decision.Decision, decision.ChangeMap
					status = &c.session.Status.Phase
				} else {
					d = c.scenario.Tick(in)
					status = &c.scenario.Status
				}
				cmd = quake.UserCmd{}
				if d.NewStep {
					c.scenarioPath = testWalkPath{}
				}
				if status.State == "running" {
					status.MovementReason = ""
				}
				if d.Place != nil {
					v := *d.Place
					if err := c.command(fmt.Sprintf("teleport %g %g %g", v[0], v[1], v[2])); err != nil {
						return err
					}
				}
				if d.Kill {
					if err := c.command("kill"); err != nil {
						return err
					}
				}
				if d.Respawn {
					cmd.Buttons = 1
				}
				if d.Push != nil {
					// Scripted actor input: real server collision/touch handlers
					// resolve the contact, independently of the bot's path guard.
					cmd = worldMove(cmd, s, d.Push[0]-s.Self[0], d.Push[1]-s.Self[1], 100, false)
					status.MovementReason = "scripted_push"
				}
				if d.Walk != nil {
					cmd, status.MovementReason = testWalkDiagnostic(s, *d.Walk, c.planner.World.Geometry, c.planner.Nav)
					if d.Route {
						cmd = c.scenarioPath.command(s, *d.Walk, c.planner.World.Geometry, c.planner.Nav)
						status.MovementReason = c.scenarioPath.reason
						if d.NewStep {
							if c.session != nil {
								c.session.RejectRoute(c.scenarioPath.reason)
							} else {
								c.scenario.RejectRoute(c.scenarioPath.reason)
							}
						}
					}
				}
				c.planner.World.Command = CommandDecision{MoveSource: "test_scenario", AimSource: "none", LimitReason: status.State}
			}
			if !safetyStop && c.testTeleportSent && c.planner.World.Map == c.testTeleportMap && c.testWalkFrames > 0 {
				age := c.testScenarioAge(frame)
				if age >= c.testWalkAfterFrames && age < c.testWalkAfterFrames+c.testWalkFrames {
					cmd = testWalkCommand(c.planner.World.Snapshot, c.testWalkTarget, c.planner.World.Geometry, c.planner.Nav)
					if c.testWalkRoute {
						cmd = c.testWalkPath.command(c.planner.World.Snapshot, c.testWalkTarget, c.planner.World.Geometry, c.planner.Nav)
					}
					c.planner.World.Command = CommandDecision{MoveSource: "test_walk", AimSource: "test_walk"}
				}
			}
			if c.testTeleportAfterSent && (c.testJumpAfterTeleportFrames > 0 &&
				jumpAge >= c.testJumpAfterTeleportFrames && jumpAge < c.testJumpAfterTeleportFrames+4 ||
				c.testJumpAgainAfterTeleportFrames > 0 && jumpAge >= c.testJumpAgainAfterTeleportFrames &&
					jumpAge < c.testJumpAgainAfterTeleportFrames+4) {
				cmd.Up = 400
				c.planner.World.Command.MoveSource = "test_jump"
			}
			if c.framePaced {
				cmd.Msec = 100
				if safetyStop {
					log.Printf("safety stop: no fresh observation for %s after frame=%d", now.Sub(c.planner.World.Updated).Truncate(time.Millisecond), frame)
				} else if c.firstMoveFrame < 0 {
					c.firstMoveFrame = frame
					c.firstMoveAt = now
				} else if frame > c.lastMoveFrame+1 {
					c.frameGaps += frame - c.lastMoveFrame - 1
				}
			}
			if cmd.Buttons&1 != 0 {
				c.attacks++
			}
			clientSequence := c.seq
			if err := c.send(quake.MovePacket(cmd, c.previous, clientSequence), false); err != nil {
				return err
			}
			if c.traceFile != nil && c.framePaced {
				entry := struct {
					Connection         int                    `json:"connection"`
					ObserverKill       bool                   `json:"test_observer_kill,omitempty"`
					ObserverRespawn    bool                   `json:"test_observer_respawn,omitempty"`
					SessionStartFrame  int                    `json:"session_start_frame,omitempty"`
					Session            *harness.SessionStatus `json:"session,omitempty"`
					Map                string                 `json:"map"`
					Spawncount         int                    `json:"spawncount"`
					EpisodeFrame       int                    `json:"episode_frame"`
					Frame              int                    `json:"frame"`
					ObservationFrame   int                    `json:"observation_frame"`
					ObservationAgeMS   int64                  `json:"observation_age_ms"`
					RelativeFrame      int                    `json:"relative_frame"`
					ClientSequence     uint32                 `json:"client_sequence"`
					Self               quake.Vec3             `json:"self"`
					SelfEntity         int                    `json:"self_entity"`
					Teammate           *quake.Vec3            `json:"teammate,omitempty"`
					TeammateEntity     int                    `json:"teammate_entity,omitempty"`
					LastTeammate       *quake.Vec3            `json:"last_teammate,omitempty"`
					TeammateAgeFrames  *int                   `json:"teammate_age_frames,omitempty"`
					Health             int16                  `json:"health"`
					Ammo               int16                  `json:"ammo"`
					Armor              int16                  `json:"armor"`
					Weapon             string                 `json:"weapon"`
					WeaponRequest      string                 `json:"weapon_request,omitempty"`
					Inventory          []quake.InventoryItem  `json:"inventory,omitempty"`
					InventoryKnown     bool                   `json:"inventory_known"`
					InventoryAgeFrames int                    `json:"inventory_age_frames"`
					OnGround           bool                   `json:"on_ground"`
					Goal               string                 `json:"goal"`
					GoalPoint          *quake.Vec3            `json:"goal_point,omitempty"`
					Route              []quake.Waypoint       `json:"route,omitempty"`
					Pickups            []quake.Object         `json:"pickups,omitempty"`
					Scenario           *harness.Status        `json:"scenario,omitempty"`
					SearchTarget       *quake.Vec3            `json:"search_target,omitempty"`
					SearchAttempt      *SearchAttempt         `json:"search_attempt,omitempty"`
					SearchRoute        *SearchRouteCheck      `json:"search_route,omitempty"`
					TeammateSound      *TeammateSoundCue      `json:"teammate_sound,omitempty"`
					TeammateMotion     *TeammateMotion        `json:"teammate_motion,omitempty"`
					TeammateEvidence   *TeammateEvidence      `json:"teammate_evidence,omitempty"`
					Navigation         string                 `json:"navigation"`
					GeometryStatus     string                 `json:"geometry_status"`
					Elevator           string                 `json:"elevator,omitempty"`
					Movers             []quake.Mover          `json:"movers,omitempty"`
					Sounds             []quake.SoundEvent     `json:"sounds,omitempty"`
					Enemies            []quake.Object         `json:"enemies,omitempty"`
					Arbitration        CommandDecision        `json:"arbitration"`
					Command            quake.UserCmd          `json:"sent_command"`
				}{
					Connection:   c.connection,
					ObserverKill: observerKill, ObserverRespawn: observerRespawn,
					SessionStartFrame: c.sessionStartFrame,
					Session:           c.sessionStatus(),
					Map:               c.planner.World.Map, Spawncount: c.spawncount, EpisodeFrame: c.moves,
					Frame: frame, ObservationFrame: c.planner.World.Snapshot.Frame,
					ObservationAgeMS: now.Sub(c.planner.World.Updated).Milliseconds(),
					RelativeFrame:    frame - c.firstMoveFrame, ClientSequence: clientSequence,
					Self: c.planner.World.Snapshot.Self, SelfEntity: c.decoder.PlayerNumber, Teammate: c.planner.World.Snapshot.Teammate,
					TeammateEntity:    c.planner.World.Snapshot.TeammateEntity,
					LastTeammate:      c.planner.World.Snapshot.LastTeammate,
					TeammateAgeFrames: c.planner.World.Snapshot.TeammateAgeFrames,
					Health:            c.planner.World.Snapshot.Health, OnGround: c.planner.World.Snapshot.OnGround,
					Ammo: c.planner.World.Snapshot.Ammo, Armor: c.planner.World.Snapshot.Armor,
					Weapon: c.planner.World.Snapshot.Weapon, WeaponRequest: weaponRequest,
					Inventory: c.planner.World.Snapshot.Inventory, InventoryKnown: c.planner.World.Snapshot.InventoryKnown, InventoryAgeFrames: c.planner.World.Snapshot.InventoryAgeFrames,
					Goal: c.planner.World.Goal, SearchTarget: c.planner.World.SearchTarget,
					Route: c.planner.World.Route, Pickups: c.planner.World.Snapshot.Pickups,
					Scenario:         c.scenarioStatus(),
					SearchAttempt:    c.planner.World.SearchAttempt,
					SearchRoute:      c.planner.World.SearchRoute,
					TeammateSound:    c.planner.World.TeammateSound,
					TeammateMotion:   c.planner.World.TeammateMotion,
					TeammateEvidence: c.planner.World.TeammateEvidence,
					Navigation:       c.planner.World.Navigation,
					GeometryStatus:   c.planner.World.GeometryStatus,
					Elevator:         c.planner.World.Elevator, Movers: c.planner.World.Snapshot.Movers,
					Sounds:      c.planner.World.Snapshot.Sounds,
					Enemies:     c.planner.World.Snapshot.Enemies,
					Arbitration: c.planner.World.Command,
					Command:     cmd,
				}
				if c.planner.hasGoal {
					goal := c.planner.goalPoint
					entry.GoalPoint = &goal
				}
				data, err := json.Marshal(entry)
				if err != nil {
					return err
				}
				if _, err = c.traceFile.Write(append(data, '\n')); err != nil {
					return err
				}
			}
			c.previous = cmd
			c.moves++
			if c.framePaced && !safetyStop {
				c.lastMoveFrame = frame
				if err := c.publishScenarioCompletion(); err != nil {
					return err
				}
			} else {
				c.nextMove = c.nextMove.Add(50 * time.Millisecond)
				if c.nextMove.Before(now.Add(-250 * time.Millisecond)) {
					c.nextMove = now.Add(50 * time.Millisecond)
				}
			}
		}
		if c.frames > 0 && c.frames%100 == 0 && now.Sub(c.lastFrame) < 100*time.Millisecond && c.moves%100 == 0 {
			w := c.planner.World
			log.Printf("map=%s frame=%d self=%v teammate=%v goal=%s nav=%s route=%d enemies=%d", w.Map, w.Snapshot.Frame, w.Snapshot.Self, w.Snapshot.Teammate, w.Goal, w.Navigation, len(w.Route), len(w.Snapshot.Enemies))
		}
	}
}
func exists(path string) bool { _, e := os.Stat(path); return e == nil }
