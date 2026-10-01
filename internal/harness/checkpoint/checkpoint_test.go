package checkpoint

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRCONEmptyAcknowledgement(t *testing.T) {
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.ParseIP("127.0.0.1")})
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	go func() {
		buf := make([]byte, 2048)
		_, addr, err := conn.ReadFromUDP(buf)
		if err == nil {
			conn.WriteToUDP([]byte("\xff\xff\xff\xffprint\n\x00"), addr)
		}
	}()
	reply, err := request(context.Background(), conn.LocalAddr().String(), "secret", "set sv_test_checkpoint_frame 0", time.Second)
	if err != nil || reply != "" {
		t.Fatalf("empty acknowledgement rejected: %q %v", reply, err)
	}
}

func TestNativeCheckpointRoundTripAndRejections(t *testing.T) {
	root := t.TempDir()
	for name, data := range map[string]string{"q2ded.exe": "engine", "baseq2/game.dll": "game", "baseq2/pak0.pak": "assets"} {
		path := filepath.Join(root, name)
		os.MkdirAll(filepath.Dir(path), 0755)
		os.WriteFile(path, []byte(data), 0600)
	}
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.ParseIP("127.0.0.1")})
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	var mu sync.Mutex
	var commands []string
	done := make(chan struct{})
	go func() {
		defer close(done)
		buf := make([]byte, 2048)
		for {
			n, addr, err := conn.ReadFromUDP(buf)
			if err != nil {
				return
			}
			cmd := strings.TrimSpace(strings.TrimRight(strings.TrimPrefix(string(buf[4:n]), "rcon secret "), "\x00"))
			mu.Lock()
			commands = append(commands, cmd)
			mu.Unlock()
			reply := ""
			switch {
			case cmd == "sv_harness_instance":
				reply = `"sv_harness_instance" is "owned"`
			case cmd == "status":
				reply = "map              : base1\n"
			case cmd == "game":
				reply = `"game" is ""`
			case cmd == "path":
				reply = "Raw search paths:\n" + filepath.ToSlash(root) + "\n\n"
			case strings.HasPrefix(cmd, "save "):
				slot := strings.TrimPrefix(cmd, "save ")
				path := filepath.Join(root, "baseq2/save", slot)
				os.MkdirAll(path, 0755)
				for _, name := range []string{"server.ssv", "game.ssv", "base1.sav", "base1.sv2", "base2.sav", "base2.sv2"} {
					os.WriteFile(filepath.Join(path, name), []byte("state-"+name), 0600)
				}
				header := make([]byte, 128)
				copy(header[32:], "*base1$start+base2")
				os.WriteFile(filepath.Join(path, "server.ssv"), header, 0600)
				reply = "Saving game...\nDone.\n"
			case strings.HasPrefix(cmd, "load "):
				reply = "Loading game...\nSavegame: " + strings.TrimPrefix(cmd, "load ") + "\n"
			}
			conn.WriteToUDP(append([]byte{255, 255, 255, 255}, []byte("print\n"+reply)...), addr)
		}
	}()
	c := Config{Version: 1, Action: "save", Server: conn.LocalAddr().String(), Instance: "owned", RuntimeRoot: root, CheckpointDir: filepath.Join(t.TempDir(), "checkpoint"), TimeoutMS: 1000}
	r, err := Run(context.Background(), c, "secret")
	if err != nil {
		t.Fatal(err)
	}
	if r.State != "saved" || r.GameplayVerified || !strings.HasPrefix(r.Slot, "harness_") {
		t.Fatalf("bad save result %+v", r)
	}
	c.Action = "load"
	r, err = Run(context.Background(), c, "secret")
	if err != nil {
		t.Fatal(err)
	}
	if r.State != "load_acknowledged" || r.GameplayVerified {
		t.Fatalf("unproved gameplay accepted %+v", r)
	}
	data, err := os.ReadFile(filepath.Join(root, "baseq2/save", r.Slot, "base2.sav"))
	if err != nil || string(data) != "state-base2.sav" {
		t.Fatal("prior campaign map omitted")
	}
	count := func() int { mu.Lock(); defer mu.Unlock(); return len(commands) }
	for _, mode := range []string{"assets", "corrupt", "missing", "symlink", "overwrite", "identity", "wrong_map"} {
		t.Run(mode, func(t *testing.T) {
			q := c
			before := count()
			restore := func() {}
			switch mode {
			case "assets":
				path := filepath.Join(root, "baseq2/pak0.pak")
				os.WriteFile(path, []byte("changed"), 0600)
				restore = func() { os.WriteFile(path, []byte("assets"), 0600) }
			case "corrupt":
				path := filepath.Join(c.CheckpointDir, "native/game.ssv")
				original, _ := os.ReadFile(path)
				os.WriteFile(path, []byte("changed"), 0600)
				restore = func() { os.WriteFile(path, original, 0600) }
			case "missing":
				path := filepath.Join(c.CheckpointDir, "native/base1.sav")
				original, _ := os.ReadFile(path)
				os.Remove(path)
				restore = func() { os.WriteFile(path, original, 0600) }
			case "symlink":
				path := filepath.Join(c.CheckpointDir, "native/extra.sav")
				if err := os.Symlink(filepath.Join(root, "q2ded.exe"), path); err != nil {
					t.Skip("symlink unavailable")
				}
				restore = func() { os.Remove(path) }
			case "overwrite":
				q.Action = "save"
			case "identity":
				q.Instance = "foreign"
			case "wrong_map":
				path := filepath.Join(c.CheckpointDir, "manifest.json")
				original, _ := os.ReadFile(path)
				os.WriteFile(path, []byte(strings.Replace(string(original), `"map": "base1"`, `"map": "base2"`, 1)), 0600)
				restore = func() { os.WriteFile(path, original, 0600) }
			}
			defer restore()
			if _, err := Run(context.Background(), q, "secret"); err == nil {
				t.Fatal("unsafe checkpoint accepted")
			}
			added := count() - before
			if mode == "identity" {
				if added != 1 {
					t.Fatal("identity check issued mutation")
				}
			} else if added != 0 {
				t.Fatal("invalid checkpoint sent server commands")
			}
		})
	}
	conn.Close()
	<-done
}

func TestCheckpointConfigRestrictsCommands(t *testing.T) {
	c := Config{Version: 1, Action: "save", Server: "127.0.0.1:29000", Instance: "owned", RuntimeRoot: "runtime", CheckpointDir: "checkpoint", TimeoutMS: 1000}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"external", "hostname", "action", "injection", "timeout"} {
		t.Run(mode, func(t *testing.T) {
			q := c
			switch mode {
			case "external":
				q.Server = "192.168.0.1:29000"
			case "hostname":
				q.Server = "localhost:29000"
			case "action":
				q.Action = "gamemap"
			case "injection":
				q.Instance = "x;quit"
			case "timeout":
				q.TimeoutMS = 0
			}
			if q.Validate() == nil {
				t.Fatal("invalid operation accepted")
			}
		})
	}
	if validateNative([]File{{Name: "server.ssv", Size: 1, SHA256: fmt.Sprintf("%064d", 0)}}, "base1") == nil {
		t.Fatal("partial native save accepted")
	}
	root := t.TempDir()
	if !portablePaths("Raw search paths:\n"+filepath.ToSlash(root)+"\n\n", root) {
		t.Fatal("portable runtime rejected")
	}
	if portablePaths("Raw search paths:\n"+filepath.ToSlash(t.TempDir())+"\n"+filepath.ToSlash(root)+"\n\n", root) {
		t.Fatal("foreign writable home directory accepted")
	}
}
