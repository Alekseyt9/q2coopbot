package main

import "q2coopbot/internal/learningenv"

func markGoalBoundary(s *learningenv.Step) error {
	// A final learned command can kill the target and hand control back to
	// rules on the next observation. Preserve that segment boundary and its
	// verified kill reward; do not reinterpret it as a regular policy step.
	return learningenv.MarkGoalBoundary(s)
}
