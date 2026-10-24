package policy

import (
	"sort"

	"q2coopbot/internal/quake"
)

const DynamicBodyObservationVersion = "observed_dynamic_bodies_v1"
const dynamicBodyRange = 512.0
const dynamicBodyCapacity = 16

// Protocol bounds describe client prediction data, not a server damage hitbox.
// A known zero solid does not imply that bullets can pass through the object.
type ObservedDynamicBody struct {
	ID                     int         `json:"id"`
	Class                  string      `json:"observed_class,omitempty"`
	ModelPath              string      `json:"observed_model,omitempty"`
	Relative               quake.Vec3  `json:"relative"`
	Distance               float64     `json:"distance"`
	Solid                  uint16      `json:"protocol_solid"`
	ProtocolBoundsKnown    bool        `json:"protocol_bounds_known"`
	ProtocolMins           *quake.Vec3 `json:"protocol_mins,omitempty"`
	ProtocolMaxs           *quake.Vec3 `json:"protocol_maxs,omitempty"`
	Animation              int         `json:"observed_animation_frame"`
	Angles                 quake.Vec3  `json:"observed_angles"`
	Skin                   uint32      `json:"observed_skin"`
	ObservedDeathAnimation bool        `json:"observed_death_animation"`
}

// Nil means unavailable; a present empty set means no eligible observed bodies.
// Legacy feature versions ignore this optional, independently versioned field.
type DynamicBodyObservation struct {
	Version   string                `json:"version"`
	Frame     int                   `json:"frame"`
	Range     float64               `json:"range"`
	Capacity  int                   `json:"capacity"`
	Truncated bool                  `json:"truncated"`
	Bodies    []ObservedDynamicBody `json:"bodies"`
}

func observedDynamicBodies(actor int, snapshot quake.Snapshot, visible func(quake.Vec3) bool) *DynamicBodyObservation {
	result := &DynamicBodyObservation{Version: DynamicBodyObservationVersion, Frame: snapshot.Frame,
		Range: dynamicBodyRange, Capacity: dynamicBodyCapacity, Bodies: []ObservedDynamicBody{}}
	objects := make(map[int]quake.Object)
	dead := make(map[int]bool)
	classes := make(map[int]string)
	for _, object := range snapshot.Enemies {
		classes[object.ID] = object.Class
	}
	for _, object := range snapshot.Barrels {
		classes[object.ID] = object.Class
	}
	for _, object := range snapshot.CollisionBodies {
		objects[object.ID] = object
	}
	for _, object := range snapshot.Defeated {
		// Defeated also includes network solid0 death animations omitted by
		// CollisionBodies. Never fill their bounds from a native/server label.
		if collision, ok := objects[object.ID]; ok {
			// Older traces omitted solid/appearance from Defeated while the
			// current collision set retained protocol bounds for the same ID.
			object.Solid = collision.Solid
			if object.ModelPath == "" {
				object.ModelPath, object.Angles, object.Skin = collision.ModelPath, collision.Angles, collision.Skin
			}
		}
		objects[object.ID], dead[object.ID], classes[object.ID] = object, true, object.Class
	}
	for id, object := range objects {
		distance := quake.Distance(snapshot.Self, object.Origin)
		if id <= 0 || id == actor || distance > dynamicBodyRange {
			continue
		}
		lo, hi, known := object.SolidBounds()
		at := object.Origin
		if known {
			at[2] += (lo[2] + hi[2]) / 2
		} else {
			at[2] += 8
		}
		if !visible(at) {
			continue
		}
		body := ObservedDynamicBody{ID: id, Class: classes[id], ModelPath: object.ModelPath,
			Relative: relative(object.Origin, snapshot.Self), Distance: distance, Solid: object.Solid,
			ProtocolBoundsKnown: known, Animation: object.Frame, Angles: object.Angles, Skin: object.Skin,
			ObservedDeathAnimation: dead[id]}
		if known {
			body.ProtocolMins, body.ProtocolMaxs = &lo, &hi
		}
		result.Bodies = append(result.Bodies, body)
	}
	sort.Slice(result.Bodies, func(i, j int) bool {
		if result.Bodies[i].Distance == result.Bodies[j].Distance {
			return result.Bodies[i].ID < result.Bodies[j].ID
		}
		return result.Bodies[i].Distance < result.Bodies[j].Distance
	})
	if len(result.Bodies) > dynamicBodyCapacity {
		result.Truncated = true
		result.Bodies = result.Bodies[:dynamicBodyCapacity]
	}
	return result
}
