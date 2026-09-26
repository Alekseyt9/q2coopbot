package harness

import "fmt"

// FrameLocation identifies a snapshot without conflating frames after a reload.
// Row is one-based in the original JSONL trace, including setup and tail frames.
type FrameLocation struct {
	Connection int    `json:"connection,omitempty"`
	Row        int    `json:"row"`
	Map        string `json:"map"`
	Generation int    `json:"generation"`
	Frame      int    `json:"frame"`
}

type TraceSegment struct {
	First FrameLocation `json:"first"`
	Last  FrameLocation `json:"last"`
	Rows  int           `json:"rows"`
}

type TimelineIssue struct {
	Kind     string         `json:"kind"`
	At       FrameLocation  `json:"at"`
	Previous *FrameLocation `json:"previous,omitempty"`
}

type TraceTimeline struct {
	Segments []TraceSegment  `json:"segments"`
	Issues   []TimelineIssue `json:"issues"`
}

type SessionTimeline struct {
	Actor TraceTimeline `json:"actor"`
	Bot   TraceTimeline `json:"bot"`
}

// verifyMapSequence requires both clients to have witnessed every generation.
// Local frame resets are expected at boundaries; generation IDs are opaque.
func verifyMapSequence(expected []string, timeline SessionTimeline) (string, *FrameLocation) {
	for _, client := range []struct {
		name  string
		trace TraceTimeline
	}{{"actor", timeline.Actor}, {"bot", timeline.Bot}} {
		if len(client.trace.Issues) > 0 {
			issue := client.trace.Issues[0]
			return client.name + ": map_sequence " + issue.Kind, &issue.At
		}
		if len(client.trace.Segments) != len(expected) {
			return fmt.Sprintf("%s: map_sequence expected %d segments, got %d", client.name, len(expected), len(client.trace.Segments)), nil
		}
		for i, segment := range client.trace.Segments {
			if segment.First.Map != expected[i] {
				location := segment.First
				return fmt.Sprintf("%s: map_sequence segment %d expected %s, got %s", client.name, i+1, expected[i], location.Map), &location
			}
		}
	}
	for i, actor := range timeline.Actor.Segments {
		bot := timeline.Bot.Segments[i]
		if actor.First.Generation != bot.First.Generation {
			location := bot.First
			return fmt.Sprintf("actor/bot map_sequence generation mismatch at segment %d", i+1), &location
		}
		if actor.First.Frame > bot.Last.Frame || bot.First.Frame > actor.Last.Frame {
			location := bot.First
			return fmt.Sprintf("actor/bot map_sequence has no shared frame at segment %d", i+1), &location
		}
	}
	return "", nil
}

// traceTimeline describes all recorded frames. Gaps in setup/tail are diagnostic;
// acceptance still requires a contiguous scenario window in analyze.
func traceTimeline(rows []Trace) TraceTimeline {
	t := TraceTimeline{Segments: []TraceSegment{}, Issues: []TimelineIssue{}}
	seen := map[int]bool{}
	var previous FrameLocation
	for i, row := range rows {
		at := FrameLocation{Row: i + 1, Map: row.Map, Generation: row.Generation, Frame: row.Frame, Connection: row.Connection}
		issue := func(kind string) {
			v := TimelineIssue{Kind: kind, At: at}
			if i > 0 {
				p := previous
				v.Previous = &p
			}
			t.Issues = append(t.Issues, v)
		}
		if row.Map == "" || row.Frame < 0 {
			issue("invalid_frame_identity")
		}
		newSegment := i == 0 || at.Map != previous.Map || at.Generation != previous.Generation || at.Connection != previous.Connection
		if i > 0 && at.Connection != previous.Connection {
			if at.Generation == previous.Generation && at.Frame <= previous.Frame {
				issue("frame_reversed_across_connection")
			}
			if at.Connection <= previous.Connection || previous.Connection == 0 {
				issue("invalid_connection_sequence")
			}
		}
		if newSegment {
			if i > 0 {
				if at.Generation == previous.Generation && at.Map != previous.Map {
					issue("map_changed_without_generation")
				} else if at.Generation != previous.Generation && seen[at.Generation] {
					issue("generation_revisited")
				}
			}
			seen[at.Generation] = true
			t.Segments = append(t.Segments, TraceSegment{First: at})
		} else if at.Frame != previous.Frame+1 {
			switch {
			case at.Frame == previous.Frame:
				issue("duplicate_frame")
			case at.Frame < previous.Frame:
				issue("frame_reversed")
			default:
				issue("frame_gap")
			}
		}
		segment := &t.Segments[len(t.Segments)-1]
		segment.Last = at
		segment.Rows++
		previous = at
	}
	return t
}
