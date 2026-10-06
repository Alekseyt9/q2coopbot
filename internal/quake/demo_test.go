package quake

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestDemoReplayAndRotation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "bot.dm2")
	recorder := &DemoRecorder{Path: path}
	decoder := NewDecoder()
	decoder.CaptureDemo = true
	signon := []byte{12}
	signon = binary.LittleEndian.AppendUint32(signon, 34)
	signon = binary.LittleEndian.AppendUint32(signon, 9)
	signon = append(signon, 0)
	signon = append(signon, []byte("baseq2\x00")...)
	signon = append(signon, 0, 0)
	signon = append(signon, []byte("Demo test\x00")...)
	signon = append(signon, 13, 33, 0)
	signon = append(signon, []byte("maps/base1.bsp\x00")...)
	// A live signon command must never be replayed as a connection command.
	signon = append(signon, 11)
	signon = append(signon, []byte("cmd configstrings 9 0\n\x00")...)
	send := func(payload []byte) {
		t.Helper()
		frames, err := decoder.Parse(payload)
		if err != nil {
			t.Fatal(err)
		}
		if err := recorder.Packet(decoder, frames); err != nil {
			t.Fatal(err)
		}
	}
	send(signon)
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("recording opened before full frame", err)
	}
	frame := []byte{20}
	frame = binary.LittleEndian.AppendUint32(frame, 1)
	frame = binary.LittleEndian.AppendUint32(frame, 0xffffffff)
	frame = append(frame, 0, 0, 17, 0, 0, 0, 0, 0, 0, 18, 0, 0)
	send(frame)
	delta := append([]byte(nil), frame...)
	binary.LittleEndian.PutUint32(delta[1:], 2)
	binary.LittleEndian.PutUint32(delta[5:], 1)
	send(delta)
	send(signon) // rotation closes and terminates the first map
	send(frame)
	if err := recorder.Close(); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{path, filepath.Join(filepath.Dir(path), "bot-002.dm2")} {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(data, []byte("cmd configstrings")) {
			t.Fatal("live command in demo")
		}
		replay := NewDecoder()
		frameCount := 0
		for {
			if len(data) < 4 {
				t.Fatal("missing terminator")
			}
			n := int(int32(binary.LittleEndian.Uint32(data)))
			data = data[4:]
			if n == -1 {
				if len(data) != 0 {
					t.Fatal("data after terminator")
				}
				break
			}
			if n < 0 || n > len(data) {
				t.Fatal("invalid record size", n)
			}
			frames, err := replay.Parse(data[:n])
			if err != nil {
				t.Fatal(err)
			}
			frameCount += len(frames)
			if replay.ServerdataSeen && (data[9] != 1 || binary.LittleEndian.Uint32(data[5:]) != 0x10009) {
				t.Fatal("wrong demo serverdata")
			}
			data = data[n:]
		}
		if frameCount < 1 || replay.Map != "base1" {
			t.Fatal("incomplete playback", frameCount, replay.Map)
		}
	}
	// A fresh process must not overwrite any existing demo.
	original, _ := os.ReadFile(path)
	next := &DemoRecorder{Path: path}
	recorder = next
	send(signon)
	send(frame)
	if err := next.Close(); err != nil {
		t.Fatal(err)
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(original, after) {
		t.Fatal("existing demo overwritten")
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(path), "bot-003.dm2")); err != nil {
		t.Fatal(err)
	}
}
