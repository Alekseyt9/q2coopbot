package main

import (
	"encoding/binary"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func demoFixture() []byte {
	signon := []byte{12}
	signon = binary.LittleEndian.AppendUint32(signon, 34)
	signon = binary.LittleEndian.AppendUint32(signon, 7)
	signon = append(signon, 1)
	signon = append(signon, []byte("baseq2\x00")...)
	signon = append(signon, 0, 0)
	signon = append(signon, []byte("Outer Base\x00")...)
	signon = append(signon, 13, 33, 0)
	signon = append(signon, []byte("maps/base1.bsp\x00")...)
	frame := []byte{20}
	frame = binary.LittleEndian.AppendUint32(frame, 1)
	frame = binary.LittleEndian.AppendUint32(frame, ^uint32(0))
	frame = append(frame, 0, 0, 17)
	frame = binary.LittleEndian.AppendUint16(frame, 128|256|2048)
	frame = append(frame, 0, 0, 88)
	frame = binary.LittleEndian.AppendUint16(frame, 0)
	frame = binary.LittleEndian.AppendUint16(frame, 16384)
	frame = binary.LittleEndian.AppendUint16(frame, 0)
	frame = append(frame, 100)
	frame = binary.LittleEndian.AppendUint32(frame, 2)
	frame = binary.LittleEndian.AppendUint16(frame, 97)
	frame = append(frame, 18, 0, 0)
	data := []byte{}
	for _, message := range [][]byte{signon, frame} {
		data = binary.LittleEndian.AppendUint32(data, uint32(len(message)))
		data = append(data, message...)
	}
	return binary.LittleEndian.AppendUint32(data, ^uint32(0))
}
func TestDemoAPIAndCache(t *testing.T) {
	root := t.TempDir()
	trace := filepath.Join(root, "bot.jsonl")
	os.WriteFile(trace, []byte("{\"map\":\"base1\",\"frame\":1,\"self\":[0,0,0]}\n"), 0600)
	path := filepath.Join(root, "bot.dm2")
	data := demoFixture()
	os.WriteFile(path, data, 0600)
	frames, complete, err := readDemo(path)
	if err != nil || !complete || len(frames) != 1 {
		t.Fatal(frames, complete, err)
	}
	f := frames[0]
	if f.Eye[2] != 22 || f.Angles[1] != 90 || f.FOV != 100 || f.Health != 97 {
		t.Fatalf("camera or HUD: %+v", f)
	}
	s := &server{root: root}
	handler := s.handler()
	request := func(url string, etag string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("GET", url, nil)
		if etag != "" {
			r.Header.Set("If-None-Match", etag)
		}
		handler.ServeHTTP(w, r)
		return w
	}
	if request("/api/demos?id=../bot.jsonl", "").Code != 404 || request("/api/demo?id=bot.jsonl&name=../bot.dm2", "").Code != 404 {
		t.Fatal("unsafe path accepted")
	}
	w := request("/api/demo?id=bot.jsonl&name=bot.dm2", "")
	if w.Code != 200 || request("/api/demo?id=bot.jsonl&name=bot.dm2", w.Header().Get("ETag")).Code != 304 {
		t.Fatal("demo HTTP/cache failed")
	}
	var response struct {
		Frames   []demoFrame
		Complete bool
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil || len(response.Frames) != 1 {
		t.Fatal(err)
	}
	os.WriteFile(path, data[:len(data)-4], 0600)
	entry, err := s.demoData(path)
	if err != nil || entry.complete {
		t.Fatal("live demo cache failed to update", err)
	}
	os.WriteFile(path, binary.LittleEndian.AppendUint32(nil, 70000), 0600)
	if _, _, err := readDemo(path); err == nil {
		t.Fatal("oversized message accepted")
	}
}
