package bot

import (
	"os"
	"path/filepath"
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
	"testing"
)

func TestMemoryRouteUnknownAndLifeReset(t *testing.T) {
	c := MemoryRouteObserver{}
	o := policy.Observation{Identity: policy.Identity{Life: 1, Frame: 10, Map: "base3"}, Health: 100, RememberedThreats: []policy.RememberedThreat{{ID: 7, AgeFrames: 1}}}
	c.Enrich(&o, nil, nil)
	if o.MemoryRoutes == nil || len(o.MemoryRoutes.Hints) != 1 || o.MemoryRoutes.Hints[0].Known {
		t.Fatal("invented missing graph")
	}
	c.cache[memoryRouteKey{Area: 1}] = cachedMemoryRoute{Found: true}
	o.Identity.Frame++
	o.Identity.Life++
	c.Enrich(&o, nil, nil)
	if len(c.cache) != 0 {
		t.Fatal("route crossed life boundary")
	}
	c.cache[memoryRouteKey{Area: 1}] = cachedMemoryRoute{Found: true}
	o.Health = 0
	c.Enrich(&o, nil, nil)
	if len(c.cache) != 0 || len(o.MemoryRoutes.Hints) != 0 {
		t.Fatal("dead route cache retained")
	}
}

func TestMemoryRouteCacheBudgetWithRealGeometry(t *testing.T) {
	root := os.Getenv("Q2_SEARCH_SCAN_ROOT")
	if root == "" {
		t.Skip("existing BSP/AAS required")
	}
	g, err := quake.LoadMap(root, "jail1")
	if err != nil {
		t.Fatal(err)
	}
	n, err := quake.LoadAAS(filepath.Join(root, "maps", "jail1.aas"))
	if err != nil {
		t.Fatal(err)
	}
	o := policy.Observation{Identity: policy.Identity{Life: 1, Map: "jail1", Frame: 120}, Health: 100, Position: quake.Vec3{-191.75, -3443.25, 88.125}}
	for i := 0; i < 8; i++ {
		o.RememberedThreats = append(o.RememberedThreats, policy.RememberedThreat{ID: 10 + i, Class: "monster_gunner", AgeFrames: 9, Relative: quake.Vec3{-129.875 + float64(i)/8, 240.5, 0}})
	}
	c := MemoryRouteObserver{}
	for i := 0; i < 4; i++ {
		before, _ := c.Statistics()
		c.Enrich(&o, n, &g)
		after, _ := c.Statistics()
		if after-before != 2 {
			t.Fatal("search budget not bounded", after-before)
		}
		o.Identity.Frame++
		for j := range o.RememberedThreats {
			o.RememberedThreats[j].AgeFrames++
		}
	}
	before, _ := c.Statistics()
	c.Enrich(&o, n, &g)
	after, hits := c.Statistics()
	if after != before || hits < 8 {
		t.Fatal("cache not reused")
	}
	for _, h := range o.MemoryRoutes.Hints {
		if !h.Known || h.ID == 10 && (!h.Found || h.Next == nil) || !h.Found && h.Next != nil {
			t.Fatal("invalid known/unsupported route", h.ID, h.Known, h.Found)
		}
	}
}
