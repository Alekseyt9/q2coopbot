package bot

import (
	"log"
	"math"

	"q2coopbot/internal/quake"
)

// elevatorRide keeps a single AAS elevator reach active across ordinary route refreshes.
type elevatorRide struct {
	model, toArea int
	gate, exit    quake.Vec3
	stage         string
}

func (p *Planner) elevatorCommand(cmd quake.UserCmd, board quake.Waypoint) quake.UserCmd {
	s := p.World.Snapshot
	if p.elevator == nil || p.elevator.model != board.Model || p.elevator.toArea != board.ToArea {
		if p.routeIndex+1 >= len(p.route) || p.route[p.routeIndex+1].ElevatorPhase != "exit" {
			p.World.Elevator = "invalid_route"
			return cmd
		}
		p.elevator = &elevatorRide{
			model: board.Model, toArea: board.ToArea, gate: board.Position,
			exit: p.route[p.routeIndex+1].Position, stage: "approach",
		}
	}
	ride := p.elevator
	setStage := func(stage string) {
		if ride.stage != stage {
			log.Printf("elevator model=%d stage=%s map=%s frame=%d", ride.model, stage, s.Map, s.Frame)
			ride.stage = stage
		}
		p.World.Elevator = ride.stage
	}
	model, ok := p.World.Geometry.Model(ride.model)
	if !ok {
		setStage("missing_model")
		return cmd
	}
	probe := quake.Vec3{
		math.Max(model.Min[0]+20, math.Min(model.Max[0]-20, ride.gate[0])),
		math.Max(model.Min[1]+20, math.Min(model.Max[1]-20, ride.gate[1])),
		ride.gate[2],
	}
	var mover *quake.Mover
	for i := range s.Movers {
		if s.Movers[i].Model == ride.model {
			mover = &s.Movers[i]
			break
		}
	}
	// A func_plat's network origin moves from the BSP model origin minus the
	// AAS rise to the model origin. Do not step onto a platform we cannot see.
	if mover == nil {
		if ride.stage == "ride" || ride.stage == "exit" || ride.stage == "landing_probe" {
			p.World.Elevator = ride.stage + "_mover_hidden"
			return cmd
		}
		if ride.stage == "approach" && quake.Horizontal(s.Self, ride.gate) > 16 {
			setStage("approach")
			return elevatorMove(cmd, s, ride.gate)
		}
		if quake.Horizontal(s.Self, probe) > 12 {
			setStage("probe_mover")
			return elevatorMove(cmd, s, probe)
		}
		setStage("waiting_for_mover")
		return cmd
	}
	center := quake.Vec3{
		(model.Min[0]+model.Max[0])/2 + mover.Origin[0],
		(model.Min[1]+model.Max[1])/2 + mover.Origin[1],
		ride.gate[2],
	}
	inside := s.Self[0] > model.Min[0]+mover.Origin[0]+12 && s.Self[0] < model.Max[0]+mover.Origin[0]-12 &&
		s.Self[1] > model.Min[1]+mover.Origin[1]+12 && s.Self[1] < model.Max[1]+mover.Origin[1]-12
	bottom := mover.Origin[2] <= model.Origin[2]-float64(board.Rise)+16
	if ride.stage == "landing_unconfirmed" {
		if s.OnGround && math.Abs(s.Self[2]-p.goalPoint[2]) < 32 && quake.Horizontal(s.Self, p.goalPoint) < 100 {
			return p.completeElevator(cmd, ride, s)
		}
		return cmd
	}
	if ride.stage == "approach" || ride.stage == "probe_mover" || ride.stage == "waiting_for_mover" {
		if inside && bottom {
			setStage("board")
		} else if quake.Horizontal(s.Self, ride.gate) > 16 {
			setStage("approach")
			return elevatorMove(cmd, s, ride.gate)
		} else {
			setStage("wait_bottom")
			return cmd
		}
	}
	if ride.stage == "wait_bottom" {
		if !bottom {
			return cmd
		}
		setStage("board")
	}
	if ride.stage == "board" {
		if !bottom && !inside {
			setStage("wait_bottom")
			return cmd
		}
		if quake.Horizontal(s.Self, center) > 20 {
			return elevatorMove(cmd, s, center)
		}
		setStage("ride")
	}
	if ride.stage == "ride" {
		// A standing player can hit the low ceiling before this tall platform
		// reaches its upper stop; duck while it carries us upward.
		cmd.Up = -200
		if !inside {
			setStage("board")
			return elevatorMove(cmd, s, center)
		}
		// The AAS exit can be below the final platform height. Begin moving
		// toward it during ascent, before the upper wall blocks a late exit.
		if mover.Origin[2] < model.Origin[2]-float64(board.Rise)+30 || !s.OnGround {
			return cmd
		}
		setStage("exit")
	}
	if ride.stage == "exit" || ride.stage == "landing_probe" {
		cmd.Up = -200
		if bottom && inside && quake.Horizontal(s.Self, ride.exit) > 32 {
			setStage("ride")
			return cmd
		}
		if ride.stage == "exit" && quake.Horizontal(s.Self, ride.exit) > 16 {
			return elevatorMove(cmd, s, ride.exit)
		}
		if s.Self[2]-p.goalPoint[2] > 40 {
			if ride.stage == "exit" && (!s.OnGround || p.Nav.AreaFor(s.Self) != ride.toArea) {
				return cmd
			}
			// An AAS area may cover both the lift edge and a lower landing.
			// Probe a short distance away from the platform, then require
			// actual ground contact at the teammate's level.
			setStage("landing_probe")
			awayX, awayY := ride.exit[0]-center[0], ride.exit[1]-center[1]
			distance := math.Hypot(awayX, awayY)
			if distance < 1 {
				setStage("landing_unconfirmed")
				return cmd
			}
			landing := quake.Vec3{ride.exit[0] + 64*awayX/distance, ride.exit[1] + 64*awayY/distance, ride.exit[2]}
			if quake.Horizontal(s.Self, landing) > 12 {
				return elevatorMove(cmd, s, landing)
			}
			setStage("landing_unconfirmed")
			return cmd
		}
		if !s.OnGround || p.Nav.AreaFor(s.Self) != ride.toArea && quake.Horizontal(s.Self, p.goalPoint) > 100 {
			return cmd
		}
		return p.completeElevator(cmd, ride, s)
	}
	return cmd
}

func (p *Planner) completeElevator(cmd quake.UserCmd, ride *elevatorRide, s quake.Snapshot) quake.UserCmd {
	p.routeIndex += 2
	p.elevator = nil
	p.World.Elevator = "completed"
	p.World.Route = p.route[p.routeIndex:]
	log.Printf("elevator model=%d completed map=%s frame=%d", ride.model, s.Map, s.Frame)
	return cmd
}

func elevatorMove(cmd quake.UserCmd, s quake.Snapshot, target quake.Vec3) quake.UserCmd {
	return worldMove(cmd, s, target[0]-s.Self[0], target[1]-s.Self[1], 300, false)
}
