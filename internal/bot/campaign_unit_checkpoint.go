package bot

import (
	"encoding/json"
	"fmt"
	"math"
	"q2coopbot/internal/quake"
	"regexp"
	"slices"
)

// Persist intention and spent budgets, never a motor command or inferred flags.
type CampaignUnitTripCheckpoint struct {
	Trip              CampaignUnitTrip `json:"trip"`
	InitialDoorOrigin quake.Vec3       `json:"initial_door_origin"`
	VerifySpent       int              `json:"verify_spent_frames"`
}

func cloneUnitTrip(t *CampaignUnitTrip) (*CampaignUnitTrip, error) {
	data, err := json.Marshal(t)
	if err != nil {
		return nil, err
	}
	var copy CampaignUnitTrip
	err = json.Unmarshal(data, &copy)
	return &copy, err
}

func captureCampaignUnitTrip(t *CampaignUnitTrip, frame int) (*CampaignUnitTripCheckpoint, error) {
	copy, err := cloneUnitTrip(t)
	if err != nil {
		return nil, err
	}
	if t.Dependency == nil {
		return nil, fmt.Errorf("unit trip dependency absent")
	}
	spent := 0
	if t.verifyStarted != 0 || t.verifyStartedSet {
		spent = frame - t.verifyStarted
	}
	return &CampaignUnitTripCheckpoint{Trip: *copy, InitialDoorOrigin: t.Dependency.initial, VerifySpent: spent}, nil
}

func validateCampaignUnitCheckpoint(c *CampaignCheckpoint, current string) error {
	name := regexp.MustCompile(`^[a-zA-Z0-9_]{1,64}$`)
	known := map[string]bool{}
	for _, m := range c.Route {
		known[m] = true
	}
	if len(c.UnitMaps) > 64 {
		return fmt.Errorf("too many checkpoint unit maps")
	}
	seen := map[string]bool{}
	for _, m := range c.UnitMaps {
		if !name.MatchString(m) || seen[m] {
			return fmt.Errorf("invalid checkpoint unit maps")
		}
		seen[m] = true
		known[m] = true
	}
	if c.UnitTrip == nil {
		return nil
	}
	u, t := c.UnitTrip, &c.UnitTrip.Trip
	if len(c.Route) < 2 || c.RouteIndex+1 >= len(c.Route) || t.OriginMap != c.Map || c.Map != c.Route[c.RouteIndex] || !known[current] || t.DoorModel <= 0 || t.Dependency == nil || t.Dependency.DoorModel != t.DoorModel || t.ElapsedFrames < 0 || t.ElapsedFrames > 1800 || t.ProbeFrames < 0 || t.ProbeFrames > 12 || u.VerifySpent < 0 || u.VerifySpent > 100 || t.EffectEvidence != "" || len(t.Stack) < 1 || len(t.Stack) > 4 {
		return fmt.Errorf("invalid unit trip checkpoint")
	}
	finite := func(v quake.Vec3) bool {
		for _, n := range v {
			if math.IsNaN(n) || math.IsInf(n, 0) || math.Abs(n) > 65536 {
				return false
			}
		}
		return true
	}
	if !finite(u.InitialDoorOrigin) || !finite(t.Dependency.ProbeFrom) || !finite(t.Dependency.ProbeTo) {
		return fmt.Errorf("invalid unit probe checkpoint")
	}
	if len(t.Stack) <= 2 && !t.Attempted || len(t.Stack) >= 3 && t.Attempted {
		return fmt.Errorf("unit attempt/stack mismatch")
	}
	kinds := []string{"verify_effect", "travel_return", "activate", "travel_outbound"}
	for i, g := range t.Stack {
		if g.Kind != kinds[i] || !known[g.Map] {
			return fmt.Errorf("invalid unit goal stack")
		}
		if i == 0 && g.Map != t.OriginMap {
			return fmt.Errorf("invalid unit return origin")
		}
		if i == 1 || i == 3 {
			if len(g.Path) < 2 || len(g.Path) > 16 || g.Leg < 0 || g.Leg >= len(g.Path)-1 || g.Path[len(g.Path)-1] != g.Map {
				return fmt.Errorf("invalid unit itinerary")
			}
			for j, m := range g.Path {
				if !known[m] || j > 0 && m == g.Path[j-1] {
					return fmt.Errorf("invalid unit itinerary map")
				}
			}
			if i == 1 && g.Path[len(g.Path)-1] != t.OriginMap || i == 3 && g.Path[0] != t.OriginMap {
				return fmt.Errorf("invalid unit itinerary origin")
			}
		}
		if i == 2 {
			a := g.Action
			if a == nil || a.Map != g.Map || a.Action != "touch" || !validUnitRoundTrip(t.OriginMap, *a) || a.Activation.Trigger.Model <= 0 || !finite(a.Activation.Trigger.Origin) {
				return fmt.Errorf("invalid unit action checkpoint")
			}
			if !slices.Equal(t.Stack[1].Path, a.ReturnMaps) || len(t.Stack) == 4 && !slices.Equal(t.Stack[3].Path, a.TravelMaps) {
				return fmt.Errorf("unit action/itinerary mismatch")
			}
			covered := false
			for _, condition := range t.Dependency.UnitConditions {
				if condition.FlagState != "unknown" {
					return fmt.Errorf("checkpoint invents cross-level flags")
				}
				if condition.RequiredFlags > 0 && condition.RequiredFlags <= 255 && a.SetsFlags&condition.RequiredFlags == condition.RequiredFlags {
					covered = true
				}
			}
			if !covered {
				return fmt.Errorf("unit checkpoint action has partial mask")
			}
		}
	}
	active := t.Stack[len(t.Stack)-1]
	if t.State != active.Kind || active.Kind == "verify_effect" && (!t.Attempted || current != t.OriginMap) || active.Kind == "activate" && current != active.Map || (active.Kind == "travel_outbound" || active.Kind == "travel_return") && current != active.Path[active.Leg] {
		return fmt.Errorf("unit goal/checkpoint map mismatch")
	}
	return nil
}

func restoreCampaignUnitTrip(u *CampaignUnitTripCheckpoint, fresh quake.Snapshot) (*CampaignUnitTrip, error) {
	t, err := cloneUnitTrip(&u.Trip)
	if err != nil {
		return nil, err
	}
	t.Dependency.initial = u.InitialDoorOrigin
	t.lastMap, t.lastFrame = fresh.Map, fresh.Frame
	if t.State == "verify_effect" {
		t.verifyStarted = fresh.Frame - u.VerifySpent
		t.verifyStartedSet = true
	}
	// Contact and movement need new native observations; elapsed budgets remain.
	return t, nil
}
