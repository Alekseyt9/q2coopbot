// Package checkpoint stores native Yamagi saves for isolated harness servers.
package checkpoint

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/netip"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

type Config struct {
	Version       int    `json:"version"`
	Action        string `json:"action"`
	Server        string `json:"server"`
	Instance      string `json:"instance"`
	RuntimeRoot   string `json:"runtime_root"`
	CheckpointDir string `json:"checkpoint_dir"`
	TimeoutMS     int    `json:"timeout_ms"`
}
type File struct {
	Name   string `json:"name"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}
type Manifest struct {
	Version    int       `json:"version"`
	CreatedUTC time.Time `json:"created_utc"`
	Map        string    `json:"map"`
	Runtime    []File    `json:"runtime"`
	Files      []File    `json:"files"`
}
type Result struct {
	Action        string `json:"action"`
	State         string `json:"state"`
	Map           string `json:"map"`
	Slot          string `json:"slot"`
	CheckpointDir string `json:"checkpoint_dir"`
	// The command acknowledgement is not proof of restored client behavior.
	GameplayVerified bool `json:"gameplay_verified"`
}

var identifier = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)
var mapLine = regexp.MustCompile(`(?m)^map\s+: ([a-zA-Z0-9_]+)\s*$`)

func (c Config) Validate() error {
	a, err := netip.ParseAddrPort(c.Server)
	if err != nil || !a.Addr().Is4() || !a.Addr().IsLoopback() || a.Port() < 1024 {
		return fmt.Errorf("checkpoint requires explicit IPv4 loopback server")
	}
	if c.Version != 1 || c.Action != "save" && c.Action != "load" || !identifier.MatchString(c.Instance) || c.RuntimeRoot == "" || c.CheckpointDir == "" || c.TimeoutMS < 100 || c.TimeoutMS > 30000 {
		return fmt.Errorf("invalid checkpoint configuration")
	}
	return nil
}

func LoadConfig(path string) (Config, error) {
	var c Config
	if err := readJSON(path, &c); err != nil {
		return c, err
	}
	if err := c.Validate(); err != nil {
		return c, err
	}
	for _, target := range []*string{&c.RuntimeRoot, &c.CheckpointDir} {
		if !filepath.IsAbs(*target) {
			*target = filepath.Join(filepath.Dir(path), *target)
		}
		abs, err := filepath.Abs(*target)
		if err != nil {
			return c, err
		}
		*target = abs
	}
	return c, nil
}
func readJSON(path string, into any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	d := json.NewDecoder(io.LimitReader(f, 2<<20))
	d.DisallowUnknownFields()
	if err = d.Decode(into); err != nil {
		return err
	}
	if d.Decode(new(any)) != io.EOF {
		return fmt.Errorf("expected one JSON document")
	}
	return nil
}
func fileRecord(root, name string) (File, error) {
	path := filepath.Join(root, filepath.FromSlash(name))
	st, err := os.Lstat(path)
	if err != nil {
		return File{}, err
	}
	if !st.Mode().IsRegular() {
		return File{}, fmt.Errorf("checkpoint requires regular files: %s", name)
	}
	f, err := os.Open(path)
	if err != nil {
		return File{}, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	return File{Name: name, Size: n, SHA256: hex.EncodeToString(h.Sum(nil))}, err
}
func runtimeRecords(root string) ([]File, error) {
	names := []string{"q2ded.exe", "baseq2/game.dll"}
	err := filepath.WalkDir(filepath.Join(root, "baseq2"), func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && d.Name() == "save" {
			return filepath.SkipDir
		}
		if !d.IsDir() {
			switch strings.ToLower(filepath.Ext(path)) {
			case ".pak", ".bsp", ".aas", ".ent":
				rel, err := filepath.Rel(root, path)
				if err != nil {
					return err
				}
				names = append(names, filepath.ToSlash(rel))
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	var records []File
	for _, name := range names {
		f, err := fileRecord(root, name)
		if err != nil {
			return nil, err
		}
		records = append(records, f)
	}
	return records, nil
}
func nativeFiles(root string) ([]File, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	var records []File
	for _, e := range entries {
		name := e.Name()
		ext := filepath.Ext(name)
		if e.IsDir() || (name != "server.ssv" && name != "game.ssv" && ext != ".sav" && ext != ".sv2") || !identifier.MatchString(strings.TrimSuffix(name, ext)) {
			return nil, fmt.Errorf("unexpected native save entry")
		}
		f, err := fileRecord(root, name)
		if err != nil {
			return nil, err
		}
		if f.Size == 0 {
			return nil, fmt.Errorf("empty native save")
		}
		records = append(records, f)
	}
	return records, nil
}
func validateNative(files []File, mapName string) error {
	if !regexp.MustCompile(`^[a-zA-Z0-9_]+$`).MatchString(mapName) {
		return fmt.Errorf("invalid saved map")
	}
	seen := map[string]bool{}
	for _, f := range files {
		if seen[f.Name] || f.Size <= 0 || len(f.SHA256) != 64 {
			return fmt.Errorf("invalid checkpoint file records")
		}
		seen[f.Name] = true
	}
	for _, name := range []string{"server.ssv", "game.ssv", mapName + ".sav", mapName + ".sv2"} {
		if !seen[name] {
			return fmt.Errorf("incomplete native checkpoint: %s", name)
		}
	}
	for name := range seen {
		if strings.HasSuffix(name, ".sav") && !seen[strings.TrimSuffix(name, ".sav")+".sv2"] || strings.HasSuffix(name, ".sv2") && !seen[strings.TrimSuffix(name, ".sv2")+".sav"] {
			return fmt.Errorf("unpaired map save")
		}
	}
	return nil
}

// Yamagi's server.ssv starts with a 32-byte comment followed by mapcmd.
// Read the saved map, not a status response from before a possible transition.
func savedMap(root string) (string, error) {
	f, err := os.Open(filepath.Join(root, "server.ssv"))
	if err != nil {
		return "", err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, 1056))
	if err != nil {
		return "", err
	}
	if len(data) < 33 {
		return "", fmt.Errorf("invalid native server header")
	}
	cmd := string(data[32:])
	end := strings.IndexByte(cmd, 0)
	if end < 0 {
		return "", fmt.Errorf("unterminated saved map command")
	}
	name := strings.SplitN(strings.SplitN(strings.TrimPrefix(cmd[:end], "*"), "+", 2)[0], "$", 2)[0]
	if !regexp.MustCompile(`^[a-zA-Z0-9_]+$`).MatchString(name) {
		return "", fmt.Errorf("invalid native saved map")
	}
	return name, nil
}
func equalFiles(a, b []File) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func portablePaths(reply, root string) bool {
	parts := strings.SplitN(reply, "Raw search paths:\n", 2)
	if len(parts) != 2 {
		return false
	}
	want, err := filepath.Abs(root)
	if err != nil {
		return false
	}
	want = filepath.ToSlash(filepath.Clean(want))
	count := 0
	for _, line := range strings.Split(parts[1], "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		got, err := filepath.Abs(filepath.FromSlash(strings.ReplaceAll(line, "\\", "/")))
		if err != nil || !strings.EqualFold(filepath.ToSlash(filepath.Clean(got)), want) {
			return false
		}
		count++
	}
	return count > 0
}
func copyFiles(source, target string, files []File) error {
	if err := os.Mkdir(target, 0755); err != nil {
		return err
	}
	for _, record := range files {
		data, err := os.ReadFile(filepath.Join(source, record.Name))
		if err != nil {
			return err
		}
		if int64(len(data)) != record.Size || fmt.Sprintf("%x", sha256.Sum256(data)) != record.SHA256 {
			return fmt.Errorf("checkpoint changed while copying")
		}
		if err = os.WriteFile(filepath.Join(target, record.Name), data, 0600); err != nil {
			return err
		}
	}
	return nil
}

// request sends once. A timeout never reissues a save/load with an unknown outcome.
func request(ctx context.Context, server, password, command string, timeout time.Duration) (string, error) {
	if password == "" || strings.ContainsAny(password, " \t\r\n\x00\";") {
		return "", fmt.Errorf("invalid Q2COOPBOT_TEST_RCON credential")
	}
	conn, err := net.Dial("udp4", server)
	if err != nil {
		return "", fmt.Errorf("checkpoint transport unavailable")
	}
	defer conn.Close()
	end := time.Now().Add(timeout)
	if deadline, ok := ctx.Deadline(); ok && deadline.Before(end) {
		end = deadline
	}
	if err = conn.SetDeadline(end); err != nil {
		return "", err
	}
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()
	packet := append([]byte{255, 255, 255, 255}, []byte("rcon "+password+" "+command+"\n")...)
	if _, err = conn.Write(packet); err != nil {
		return "", fmt.Errorf("checkpoint request failed")
	}
	var reply strings.Builder
	buf := make([]byte, 65536)
	for {
		n, err := conn.Read(buf)
		if err != nil {
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			if reply.Len() > 0 {
				return reply.String(), nil
			}
			return "", fmt.Errorf("checkpoint response unavailable; command outcome unknown")
		}
		if n < 10 || string(buf[:4]) != "\xff\xff\xff\xff" || string(buf[4:10]) != "print\n" {
			continue
		}
		reply.WriteString(strings.TrimRight(string(buf[10:n]), "\x00"))
		if reply.Len() > 1<<20 {
			return "", fmt.Errorf("checkpoint response too large")
		}
		// RCON redirects can span several datagrams. Drain only this response.
		quiet := time.Now().Add(150 * time.Millisecond)
		if quiet.After(end) {
			quiet = end
		}
		conn.SetReadDeadline(quiet)
	}
}

func Run(ctx context.Context, c Config, password string) (Result, error) {
	result := Result{Action: c.Action, CheckpointDir: c.CheckpointDir}
	if err := c.Validate(); err != nil {
		return result, err
	}
	runtime, err := runtimeRecords(c.RuntimeRoot)
	if err != nil {
		return result, err
	}
	var manifest Manifest
	if c.Action == "load" {
		if err = readJSON(filepath.Join(c.CheckpointDir, "manifest.json"), &manifest); err != nil {
			return result, err
		}
		if manifest.Version != 1 || !equalFiles(runtime, manifest.Runtime) {
			return result, fmt.Errorf("checkpoint runtime/assets mismatch")
		}
		files, err := nativeFiles(filepath.Join(c.CheckpointDir, "native"))
		if err != nil {
			return result, err
		}
		if !equalFiles(files, manifest.Files) {
			return result, fmt.Errorf("checkpoint integrity mismatch")
		}
		if err = validateNative(files, manifest.Map); err != nil {
			return result, err
		}
		nativeMap, err := savedMap(filepath.Join(c.CheckpointDir, "native"))
		if err != nil || nativeMap != manifest.Map {
			return result, fmt.Errorf("native map differs from checkpoint manifest")
		}
	} else if _, err = os.Stat(c.CheckpointDir); !os.IsNotExist(err) {
		return result, fmt.Errorf("save requires a new checkpoint directory")
	}
	timeout := time.Duration(c.TimeoutMS) * time.Millisecond
	reply, err := request(ctx, c.Server, password, "sv_harness_instance", timeout)
	if err != nil {
		return result, err
	}
	if !strings.Contains(reply, `"sv_harness_instance" is "`+c.Instance+`"`) {
		return result, fmt.Errorf("isolated server identity mismatch")
	}
	reply, err = request(ctx, c.Server, password, "game", timeout)
	if err != nil {
		return result, err
	}
	if !strings.Contains(reply, `"game" is ""`) && !strings.Contains(reply, `"game" is "baseq2"`) {
		return result, fmt.Errorf("checkpoint requires the baseq2 game")
	}
	reply, err = request(ctx, c.Server, password, "path", timeout)
	if err != nil {
		return result, err
	}
	if !portablePaths(reply, c.RuntimeRoot) {
		return result, fmt.Errorf("server paths do not match the isolated portable runtime")
	}
	var nonce [12]byte
	if _, err = rand.Read(nonce[:]); err != nil {
		return result, err
	}
	slot := "harness_" + hex.EncodeToString(nonce[:])
	result.Slot = slot
	if c.Action == "save" {
		reply, err = request(ctx, c.Server, password, "save "+slot, timeout)
		if err != nil {
			return result, err
		}
		if !strings.Contains(reply, "Saving game...") || !strings.Contains(reply, "Done.") || strings.Contains(reply, "Couldn't") {
			return result, fmt.Errorf("native save rejected or incomplete")
		}
		root := filepath.Join(c.RuntimeRoot, "baseq2", "save", slot)
		manifest.Map, err = savedMap(root)
		if err != nil {
			return result, err
		}
		files, err := nativeFiles(root)
		if err != nil {
			return result, fmt.Errorf("native files absent; server must use this portable runtime")
		}
		if err = validateNative(files, manifest.Map); err != nil {
			return result, err
		}
		if err = os.MkdirAll(filepath.Dir(c.CheckpointDir), 0755); err != nil {
			return result, err
		}
		if err = os.Mkdir(c.CheckpointDir, 0755); err != nil {
			return result, err
		}
		if err = copyFiles(root, filepath.Join(c.CheckpointDir, "native"), files); err != nil {
			return result, err
		}
		manifest.Version = 1
		manifest.CreatedUTC = time.Now().UTC()
		manifest.Runtime = runtime
		manifest.Files = files
		data, err := json.MarshalIndent(manifest, "", "  ")
		if err != nil {
			return result, err
		}
		if err = os.WriteFile(filepath.Join(c.CheckpointDir, "manifest.json"), data, 0600); err != nil {
			return result, err
		}
		result.Map = manifest.Map
		result.State = "saved"
		return result, nil
	}
	parent := filepath.Join(c.RuntimeRoot, "baseq2", "save")
	if err = os.MkdirAll(parent, 0755); err != nil {
		return result, err
	}
	if err = copyFiles(filepath.Join(c.CheckpointDir, "native"), filepath.Join(parent, slot), manifest.Files); err != nil {
		return result, err
	}
	reply, err = request(ctx, c.Server, password, "load "+slot, timeout)
	if err != nil {
		return result, err
	}
	if !strings.Contains(reply, "Savegame: "+slot) || strings.Contains(reply, "No such") || strings.Contains(reply, "Couldn't") {
		return result, fmt.Errorf("native load not acknowledged")
	}
	reply, err = request(ctx, c.Server, password, "status", timeout)
	if err != nil {
		return result, err
	}
	match := mapLine.FindStringSubmatch(reply)
	if len(match) != 2 || match[1] != manifest.Map {
		return result, fmt.Errorf("loaded map not confirmed")
	}
	result.Map = manifest.Map
	result.State = "load_acknowledged"
	return result, nil
}
