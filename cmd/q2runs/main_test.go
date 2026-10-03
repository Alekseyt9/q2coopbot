package main

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestPaginationSearchAndTraceSafety(t *testing.T) {
	root := t.TempDir()
	for i := 0; i < 31; i++ {
		path := filepath.Join(root, fmt.Sprintf("run-%02d.jsonl", i))
		data := fmt.Sprintf(`{"map":"base%d","frame":10,"self":[1,2,3]}`+"\n", i%2+1)
		if e := os.WriteFile(path, []byte(data), 0600); e != nil {
			t.Fatal(e)
		}
		stamp := time.Unix(int64(100+i), 0)
		os.Chtimes(path, stamp, stamp)
	}
	os.WriteFile(filepath.Join(root, "tests.jsonl"), []byte(`{"Action":"pass"}`), 0600)
	s := &server{root: root}
	h := s.handler()
	request := func(url string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("GET", url, nil))
		return w
	}
	var result struct {
		Items              []run `json:"items"`
		Total, Page, Pages int
	}
	w := request("/api/runs?page=2&size=25")
	if e := json.Unmarshal(w.Body.Bytes(), &result); e != nil {
		t.Fatal(e)
	}
	if result.Total != 31 || result.Page != 2 || result.Pages != 2 || len(result.Items) != 6 || result.Items[0].ID != "run-05.jsonl" {
		t.Fatalf("unexpected page: %+v", result)
	}
	w = request("/api/runs?q=base2&size=100")
	json.Unmarshal(w.Body.Bytes(), &result)
	if result.Total != 15 || result.Items[0].ID != "run-29.jsonl" {
		t.Fatalf("unexpected search: %+v", result)
	}
	if request("/api/trace?id=../outside.jsonl").Code != 404 {
		t.Fatal("unlisted path accepted")
	}
	if request("/api/trace?id=run-30.jsonl").Code != 200 {
		t.Fatal("valid trace rejected")
	}
	w = request("/api/runs?page=999999999&size=100")
	json.Unmarshal(w.Body.Bytes(), &result)
	if result.Page != 1 || len(result.Items) != 31 {
		t.Fatal("page clamp failed")
	}
}

func TestPartialLiveTrace(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.jsonl")
	os.WriteFile(path, []byte("{\"map\":\"base1\",\"self\":[0,0,0],\"frame\":1}\n{\"map\":"), 0600)
	points, skipped, e := readTrace(path, false)
	if e != nil || len(points) != 1 || skipped != 1 {
		t.Fatalf("points=%v skipped=%d error=%v", points, skipped, e)
	}
}
