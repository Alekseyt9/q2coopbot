package bot

import (
	"q2coopbot/internal/policy"
	"q2coopbot/internal/quake"
)

type memoryRouteKey struct {
	Area int
	Goal quake.Vec3
}
type cachedMemoryRoute struct {
	Path  []quake.Waypoint
	Found bool
	Next  int
}
type MemoryRouteObserver struct {
	identity policy.Identity
	nav      *quake.Navigator
	geometry *quake.MapInfo
	cache    map[memoryRouteKey]cachedMemoryRoute
	searches int
	hits     int
}

// Enrich never touches Planner goals, routes, commands, or hidden entities.
// At most two graph searches per observation; cached geometry remains static.
func (c *MemoryRouteObserver) Enrich(o *policy.Observation, n *quake.Navigator, g *quake.MapInfo) {
	if c.cache == nil || !policy.SameLife(c.identity, o.Identity) || o.Identity.Frame < c.identity.Frame || o.Identity.Frame > c.identity.Frame+1 || c.nav != n || c.geometry != g {
		*c = MemoryRouteObserver{cache: make(map[memoryRouteKey]cachedMemoryRoute), nav: n, geometry: g}
	}
	c.identity = o.Identity
	r := &policy.MemoryRouteObservation{Version: policy.MemoryRouteObservationVersion, Frame: o.Identity.Frame, Hints: []policy.MemoryRouteHint{}}
	o.MemoryRoutes = r
	if o.Health <= 0 || o.AgeMS < 0 || o.AgeMS > 300 {
		clear(c.cache)
		return
	}
	if len(o.Enemies) > 0 {
		return
	}
	budget := 2
	for i, t := range o.RememberedThreats {
		if i >= 8 {
			break
		}
		h := policy.MemoryRouteHint{ID: t.ID, Age: t.AgeFrames}
		if t.AgeFrames < 0 || t.AgeFrames > policy.ThreatMemoryFrames || n == nil || g == nil || !g.MovementComplete() {
			r.Hints = append(r.Hints, h)
			continue
		}
		goal := o.Position
		for axis := range goal {
			goal[axis] += t.Relative[axis]
		}
		key := memoryRouteKey{n.AreaFor(o.Position), goal}
		entry, ok := c.cache[key]
		if !ok && budget > 0 {
			if len(c.cache) >= 128 {
				clear(c.cache)
			}
			entry.Path, entry.Found = n.SearchRoute(o.Position, goal)
			c.cache[key] = entry
			budget--
			c.searches++
		} else if ok {
			c.hits++
		}
		if !ok && budget == 0 {
			if _, ready := c.cache[key]; !ready {
				r.Hints = append(r.Hints, h)
				continue
			}
		}
		h.Known = true
		h.Found = entry.Found
		h.Direct = g.PlayerMoveClear(o.Position, goal)
		if entry.Found {
			for entry.Next < len(entry.Path) && quake.Horizontal(o.Position, entry.Path[entry.Next].Position) < 24 {
				entry.Next++
			}
			at := goal
			if entry.Next < len(entry.Path) {
				at = entry.Path[entry.Next].Position
				h.Kind = entry.Path[entry.Next].Kind
			}
			previous := o.Position
			for _, p := range entry.Path[entry.Next:] {
				h.Distance += quake.Horizontal(previous, p.Position)
				previous = p.Position
			}
			h.Distance += quake.Horizontal(previous, goal)
			relative := at
			for axis := range relative {
				relative[axis] -= o.Position[axis]
			}
			h.Next = &relative
			c.cache[key] = entry
		}
		r.Hints = append(r.Hints, h)
	}
}

func (c *MemoryRouteObserver) Statistics() (searches, hits int) { return c.searches, c.hits }
