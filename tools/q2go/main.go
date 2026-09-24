package main

import (
	"context"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

//go:embed chktbl.hex
var tableFile embed.FS
var checkTable [1024]byte

func init() {
	raw, _ := tableFile.ReadFile("chktbl.hex")
	v, e := hex.DecodeString(strings.TrimSpace(string(raw)))
	if e != nil || len(v) != 960 {
		panic("invalid command checksum table")
	}
	copy(checkTable[:], v)
}

type UserCmd struct {
	Pitch, Yaw, Roll, Forward, Side, Up int16
	Buttons, Impulse, Msec, Light       byte
}

func deltaCmd(a, b UserCmd) []byte {
	out := []byte{0}
	fields := [][2]int16{{a.Pitch, b.Pitch}, {a.Yaw, b.Yaw}, {a.Roll, b.Roll}, {a.Forward, b.Forward}, {a.Side, b.Side}, {a.Up, b.Up}}
	for i, p := range fields {
		if p[0] != p[1] {
			out[0] |= 1 << i
			out = binary.LittleEndian.AppendUint16(out, uint16(p[1]))
		}
	}
	if a.Buttons != b.Buttons {
		out[0] |= 64
		out = append(out, b.Buttons)
	}
	if a.Impulse != b.Impulse {
		out[0] |= 128
		out = append(out, b.Impulse)
	}
	return append(out, b.Msec, b.Light)
}
func crcMove(data []byte, seq uint32) byte {
	n := min(len(data), 60)
	input := append([]byte{}, data[:n]...)
	offset := int(seq % 1020)
	input = append(input, checkTable[offset:offset+4]...)
	crc := uint16(0xffff)
	sum := 0
	for _, b := range input {
		sum += int(b)
		crc ^= uint16(b) << 8
		for i := 0; i < 8; i++ {
			if crc&0x8000 != 0 {
				crc = crc<<1 ^ 0x1021
			} else {
				crc <<= 1
			}
		}
	}
	return byte(int(crc) ^ sum)
}
func movePacket(cmd, prev UserCmd, seq uint32) []byte {
	out := []byte{2, 0}
	out = binary.LittleEndian.AppendUint32(out, ^uint32(0))
	out = append(out, deltaCmd(UserCmd{}, prev)...)
	out = append(out, deltaCmd(prev, prev)...)
	out = append(out, deltaCmd(prev, cmd)...)
	out[1] = crcMove(out[2:], seq)
	return out
}

type World struct {
	Map            string            `json:"map"`
	Geometry       *MapInfo          `json:"geometry,omitempty"`
	AASLoaded      bool              `json:"aas_loaded"`
	Areas          int               `json:"areas"`
	Reachabilities int               `json:"reachabilities"`
	Navigation     string            `json:"navigation"`
	Goal           string            `json:"goal"`
	Strategy       *StrategyDecision `json:"strategy,omitempty"`
	Route          []Waypoint        `json:"route,omitempty"`
	Snapshot       Snapshot          `json:"snapshot"`
	Updated        time.Time         `json:"updated"`
}
type Planner struct {
	Nav          *Navigator
	AASDir       string
	World        World
	lastSelf     Vec3
	lastProgress time.Time
	routeAt      time.Time
	target       Vec3
	failures     int
	goalPoint    Vec3
	hasGoal      bool
	decision     *StrategyDecision
}

func (p *Planner) applyDecision(d StrategyDecision) {
	p.decision = &d
	log.Printf("system2 choice=%s latency=%dms reason=%q", d.Choice, d.LatencyMS, d.Reason)
}

func (p *Planner) setMap(name, root string) {
	if p.World.Map == name {
		return
	}
	p.World = World{Map: name, Navigation: "aas_missing"}
	p.Nav = nil
	p.failures = 0
	p.decision = nil
	if name == "" {
		return
	}
	if info, e := LoadMap(root, name); e == nil {
		p.World.Geometry = &info
		log.Printf("map=%s BSP entities=%d brushes=%d", name, len(info.Entities), info.Brushes)
	} else {
		log.Printf("map=%s BSP unavailable: %v", name, e)
	}
	paths := []string{filepath.Join(p.AASDir, name+".aas"), filepath.Join(root, "maps", name+".aas")}
	var n *Navigator
	var e error
	for _, path := range paths {
		candidate, loadErr := LoadAAS(path)
		if loadErr != nil {
			e = loadErr
			continue
		}
		edges := 0
		for _, list := range candidate.Edges {
			edges += len(list)
		}
		if edges == 0 {
			e = fmt.Errorf("AAS contains no traversable reaches: %s", path)
			continue
		}
		n = candidate
		break
	}
	if n == nil {
		log.Printf("map=%s AAS unavailable: %v", name, e)
		return
	}
	p.Nav = n
	p.World.AASLoaded = true
	p.World.Areas = len(n.Areas)
	for _, edges := range n.Edges {
		p.World.Reachabilities += len(edges)
	}
	p.World.Navigation = "ready"
	log.Printf("map=%s AAS areas=%d reachabilities=%d", name, p.World.Areas, p.World.Reachabilities)
}
func (p *Planner) update(s Snapshot, root string) {
	p.setMap(s.Map, root)
	if p.World.Geometry != nil && p.World.Geometry.collision != nil {
		for i := range s.Enemies {
			from, to := s.Self, s.Enemies[i].Origin
			from[2] += 22
			to[2] += 22
			clear := p.World.Geometry.collision.ClearShot(from, to)
			s.Enemies[i].ClearShot = &clear
		}
	}
	p.World.Snapshot = s
	p.World.Updated = time.Now()
	if p.decision != nil && time.Since(p.decision.At) < 8*time.Second {
		p.World.Strategy = p.decision
	} else {
		p.World.Strategy = nil
	}
	p.World.Goal = "wait_for_teammate"
	p.World.Route = nil
	p.hasGoal = false
	if s.Teammate == nil {
		return
	}
	goal := *s.Teammate
	p.hasGoal = true
	for _, pickup := range s.Pickups {
		if s.Health < 45 && strings.Contains(pickup.Class, "health") && horizontal(s.Self, pickup.Origin) < 300 {
			goal = pickup.Origin
			p.World.Goal = "recover_health"
			break
		}
	}
	if p.World.Goal == "recover_health" { /* keep health objective */
	} else if horizontal(s.Self, goal) < 100 {
		p.World.Goal = "cover_teammate"
	} else {
		p.World.Goal = "follow_teammate"
	}
	if p.World.Strategy != nil {
		switch p.World.Strategy.Choice {
		case "recover":
			for _, pickup := range s.Pickups {
				if strings.Contains(pickup.Class, "health") && horizontal(s.Self, pickup.Origin) < 300 {
					goal = pickup.Origin
					p.World.Goal = "recover_health"
					break
				}
			}
		case "engage":
			for _, enemy := range s.Enemies {
				if enemy.ClearShot != nil && *enemy.ClearShot {
					p.World.Goal = "engage_enemy"
					break
				}
			}
		case "reposition":
			if p.World.Goal != "recover_health" {
				p.World.Goal = "follow_teammate"
			}
		}
	}
	p.goalPoint = goal
	if p.Nav == nil {
		if p.World.Geometry != nil && p.World.Geometry.collision != nil && horizontal(s.Self, goal) < 256 && math.Abs(s.Self[2]-goal[2]) < 64 {
			from, to := s.Self, goal
			from[2] += 18
			to[2] += 18
			if p.World.Geometry.collision.ClearShot(from, to) {
				p.World.Navigation = "direct_clear"
			}
		}
		return
	}
	if horizontal(goal, p.target) > 96 || time.Since(p.routeAt) > 2*time.Second {
		p.routeAt = time.Now()
		p.target = goal
		route, ok := p.Nav.Route(s.Self, goal)
		if ok {
			p.World.Route = route
			p.World.Navigation = "ready"
		} else {
			p.World.Navigation = "unreachable"
		}
	} else {
		route, ok := p.Nav.Route(s.Self, goal)
		if ok {
			p.World.Route = route
		} else {
			p.World.Navigation = "unreachable"
		}
	}
	if horizontal(s.Self, p.lastSelf) > 48 {
		p.lastProgress = time.Now()
		p.lastSelf = s.Self
		p.failures = 0
	} else if p.lastProgress.IsZero() {
		p.lastProgress = time.Now()
		p.lastSelf = s.Self
	}
	if p.World.Navigation == "unreachable" && p.World.Geometry != nil && p.World.Geometry.collision != nil && horizontal(s.Self, goal) < 256 && math.Abs(s.Self[2]-goal[2]) < 64 {
		from, to := s.Self, goal
		from[2] += 18
		to[2] += 18
		if p.World.Geometry.collision.ClearShot(from, to) {
			p.World.Navigation = "direct_clear"
		}
	}
}
func (p *Planner) command(prev UserCmd) UserCmd {
	cmd := UserCmd{Yaw: prev.Yaw, Msec: 50}
	s := p.World.Snapshot
	if p.World.Map == "" || s.Frame == 0 {
		return cmd
	}
	if s.Health <= 0 {
		return cmd
	}
	var enemy *Object
	best := math.Inf(1)
	for i := range s.Enemies {
		d := distance(s.Self, s.Enemies[i].Origin)
		if d < best {
			best = d
			enemy = &s.Enemies[i]
		}
	}
	if enemy != nil && best < 650 && (s.Ammo > 0 || strings.Contains(strings.ToLower(s.Weapon), "blast")) && enemy.ClearShot != nil && *enemy.ClearShot {
		from, to := s.Self, enemy.Origin
		from[2] += 22
		to[2] += 22
		cmd.Yaw = yawTo(from, to, s.DeltaAngles[1])
		cmd.Pitch = pitchTo(from, to, s.DeltaAngles[0])
		cmd.Buttons = 1
	}
	if p.World.Goal != "follow_teammate" && p.World.Goal != "recover_health" || p.World.Navigation != "ready" && p.World.Navigation != "direct_clear" || !p.hasGoal {
		return cmd
	}
	target := p.goalPoint
	jump := false
	for _, wp := range p.World.Route {
		if horizontal(s.Self, wp.Position) > 10 || math.Abs(s.Self[2]-wp.Position[2]) > 64 {
			target = wp.Position
			jump = wp.Jump
			break
		}
	}
	if horizontal(s.Self, target) < 10 {
		return cmd
	}
	if cmd.Buttons == 0 {
		cmd.Yaw = yawTo(s.Self, target, s.DeltaAngles[1])
	}
	cmd.Forward = 400
	if jump || target[2]-s.Self[2] > 32 {
		cmd.Up = 200
	}
	if time.Since(p.lastProgress) > 3*time.Second {
		p.failures++
		p.lastProgress = time.Now()
		cmd.Up = 200
		if p.failures%2 == 1 {
			cmd.Side = 300
		} else {
			cmd.Side = -300
		}
	}
	return cmd
}

type Client struct {
	conn             *net.UDPConn
	address          *net.UDPAddr
	seq, serverSeq   uint32
	serverReliable   uint32
	qport            uint16
	challenge        int
	connected, begun bool
	spawncount       int
	lastHandshake    string
	handshakeAt      time.Time
	decoder          *Decoder
	planner          *Planner
	root             string
	previous         UserCmd
	nextMove         time.Time
	lastFrame        time.Time
	frames, moves    int
	attacks          int
	worldFile        string
	stopFile         string
	name             string
	strategist       *Strategist
	seenCommands     map[string]bool
	duration         time.Duration
	start            time.Time
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
func (c *Client) handle(packet []byte) {
	if len(packet) >= 4 && string(packet[:4]) == "\xff\xff\xff\xff" {
		message := string(packet[4:])
		log.Printf("server OOB: %q", strings.TrimSpace(message))
		if strings.Contains(message, "challenge ") && !c.connected {
			re := regexp.MustCompile(`challenge\s+(-?\d+)`)
			m := re.FindStringSubmatch(message)
			if len(m) > 1 {
				c.challenge, _ = strconv.Atoi(m[1])
				userinfo := fmt.Sprintf(`\name\%s\skin\male/grunt\rate\25000\msg\1\hand\2\fov\90`, c.name)
				_ = c.oob(fmt.Sprintf("connect 34 %d %d %q\n", c.qport, c.challenge, userinfo))
			}
		}
		if strings.Contains(message, "client_connect") && !c.connected {
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
	if len(payload) >= 9 && payload[0] == 12 {
		c.spawncount = int(int32(binary.LittleEndian.Uint32(payload[5:])))
		c.begun = false
		c.previous = UserCmd{}
		c.planner.setMap("", c.root)
		c.seenCommands = map[string]bool{}
		command := fmt.Sprintf("configstrings %d 0", c.spawncount)
		c.seenCommands["cmd "+command] = true
		_ = c.command(command)
	}
	frames, e := c.decoder.Parse(payload)
	if e != nil {
		c.decoder.Errors++
		c.decoder.LastError = e.Error()
	}
	for _, f := range frames {
		c.frames++
		c.lastFrame = time.Now()
		s := c.decoder.Snapshot(f)
		if s.Map != "" {
			c.planner.update(s, c.root)
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
			c.seenCommands = map[string]bool{}
			c.planner.setMap("", c.root)
		case request == "reconnect":
			c.begun = false
			c.seenCommands = map[string]bool{}
			_ = c.command("new")
		case strings.HasPrefix(request, "cmd configstrings "), strings.HasPrefix(request, "cmd baselines "):
			_ = c.command(strings.TrimPrefix(request, "cmd "))
		case strings.HasPrefix(request, "precache "):
			parts := strings.Fields(request)
			if len(parts) > 1 {
				c.begun = true
				c.nextMove = time.Now()
				_ = c.command("begin " + parts[1])
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
func (c *Client) run(ctx context.Context) error {
	c.start = time.Now()
	c.nextMove = c.start
	c.handshakeAt = c.start
	_ = c.oob("getchallenge\n")
	buffer := make([]byte, 65535)
	for {
		if ctx.Err() != nil || c.duration > 0 && time.Since(c.start) >= c.duration || c.stopFile != "" && exists(c.stopFile) {
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
		if c.connected && !c.begun && c.lastHandshake != "" && now.Sub(c.handshakeAt) >= 2*time.Second {
			_ = c.command(c.lastHandshake)
		}
		if c.begun && now.After(c.nextMove) {
			cmd := c.planner.command(c.previous)
			if cmd.Buttons&1 != 0 {
				c.attacks++
			}
			_ = c.send(movePacket(cmd, c.previous, c.seq), false)
			c.previous = cmd
			c.moves++
			c.nextMove = c.nextMove.Add(50 * time.Millisecond)
			if c.nextMove.Before(now.Add(-250 * time.Millisecond)) {
				c.nextMove = now.Add(50 * time.Millisecond)
			}
		}
		if c.frames > 0 && c.frames%100 == 0 && now.Sub(c.lastFrame) < 100*time.Millisecond && c.moves%100 == 0 {
			w := c.planner.World
			log.Printf("map=%s frame=%d self=%v teammate=%v goal=%s nav=%s route=%d enemies=%d", w.Map, w.Snapshot.Frame, w.Snapshot.Self, w.Snapshot.Teammate, w.Goal, w.Navigation, len(w.Route), len(w.Snapshot.Enemies))
		}
	}
}
func exists(path string) bool { _, e := os.Stat(path); return e == nil }
func main() {
	host := flag.String("host", "127.0.0.1", "server host")
	port := flag.Int("port", 27910, "server UDP port")
	name := flag.String("name", "GoCoopMate", "client name")
	root := flag.String("game-dir", "", "baseq2 directory containing maps/*.aas")
	aasDir := flag.String("aas-dir", "", "directory containing campaign AAS files; default game-dir/maps")
	duration := flag.Duration("duration", 0, "session duration; 0 runs until Ctrl-C")
	worldFile := flag.String("world-json", "", "current world state output")
	stopFile := flag.String("stop-file", "", "disconnect when file appears")
	system2Model := flag.String("system2-model", "", "optional Ollama strategy model, called asynchronously every 2 seconds")
	flag.Parse()
	if *root == "" {
		log.Fatal("--game-dir is required")
	}
	if *aasDir == "" {
		*aasDir = filepath.Join(*root, "maps")
	}
	address, e := net.ResolveUDPAddr("udp4", fmt.Sprintf("%s:%d", *host, *port))
	if e != nil {
		log.Fatal(e)
	}
	conn, e := net.ListenUDP("udp4", nil)
	if e != nil {
		log.Fatal(e)
	}
	defer conn.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	client := &Client{conn: conn, address: address, qport: uint16(rand.Intn(65535) + 1), seq: 1, decoder: NewDecoder(), planner: &Planner{AASDir: *aasDir}, root: *root, worldFile: *worldFile, stopFile: *stopFile, name: *name, duration: *duration}
	if *system2Model != "" {
		client.strategist = NewStrategist(*system2Model)
	}
	if e = client.run(ctx); e != nil {
		log.Fatal(e)
	}
	if *worldFile != "" {
		_ = client.writeWorld()
	}
	log.Printf("finished connected=%t begun=%t frames=%d moves=%d attacks=%d decode_errors=%d last_error=%q", client.connected, client.begun, client.frames, client.moves, client.attacks, client.decoder.Errors, client.decoder.LastError)
}
