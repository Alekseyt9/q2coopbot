package compare

import "testing"

func suite() Suite {
	yes := true
	return Suite{Provenance: true, Manifest: Manifest{Version: 1, SourceUnchanged: true, ArtifactsUnchanged: true, Artifacts: []File{{"q2scenario-report.exe", "a"}, {"q2coopbot.exe", "b"}}}, Results: []Run{{Fixture: "fixture", Accepted: &yes, Speed: map[string]any{"wall_seconds": float64(10)}}}}
}
func TestComparisonSeparatesCompatibilityFromOutcomes(t *testing.T) {
	a, b := suite(), suite()
	no := false
	b.Results[0].Accepted = &no
	b.Results[0].Speed["wall_seconds"] = float64(12)
	r := Compare(a, b)
	if !r.Compatible || r.Groups[0].AfterAccepted != 0 || *r.Groups[0].Metrics["speed.wall_seconds"].Delta != 2 {
		t.Fatalf("%+v", r)
	}
	b.Manifest.Artifacts[0].Hash = "different"
	r = Compare(a, b)
	if r.Compatible || len(r.Groups[0].Metrics) != 0 {
		t.Fatalf("mixed analyzer metrics: %+v", r)
	}
}
func TestComparisonMissingDataIsNotZero(t *testing.T) {
	a, b := suite(), suite()
	b.Results[0].Speed = nil
	r := Compare(a, b)
	m := r.Groups[0].Metrics["speed.wall_seconds"]
	if m.After != nil || m.Delta != nil {
		t.Fatal(m)
	}
	b.Results[0].Fixture = "other"
	r = Compare(a, b)
	if r.Compatible || len(r.Groups) != 2 {
		t.Fatal(r)
	}
	for _, g := range r.Groups {
		if len(g.Metrics) != 0 {
			t.Fatal("incompatible fixtures combined")
		}
	}
}
func TestSummaryVariationAndClientChange(t *testing.T) {
	a, b := suite(), suite()
	v := b.Results[0]
	v.Speed = map[string]any{"wall_seconds": float64(20)}
	b.Results = append(b.Results, v)
	b.Manifest.Artifacts[1].Hash = "new-client"
	r := Compare(a, b)
	s := r.Groups[0].Metrics["speed.wall_seconds"].After
	if !r.Compatible || !r.ClientChanged || s.Count != 2 || s.Min != 10 || s.Max != 20 || s.Mean != 15 {
		t.Fatal(r, s)
	}
	a.Provenance = false
	if Compare(a, b).Compatible {
		t.Fatal("invalid provenance accepted")
	}
}
