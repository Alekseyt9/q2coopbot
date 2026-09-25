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
		if ride.stage == "ride" || ride.stage == "exit" {
			p.World.Elevator = ride.stage + "_mover_hidden"
			return cmd
		}
		if quake.Horizontal(s.Self, ride.gate) > 16 {
			setStage("approach")
			return elevatorMove(cmd, s, ride.gate)
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
	top := mover.Origin[2] >= model.Origin[2]-12
	if ride.stage == "approach" || ride.stage == "waiting_for_mover" {
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
		if !inside && !top {
			setStage("board")
			return elevatorMove(cmd, s, center)
		}
		if !top || s.Self[2] < ride.exit[2]-24 || !s.OnGround {
			return cmd
		}
		setStage("exit")
	}
	if ride.stage == "exit" {
		if !top && inside {
			setStage("ride")
			return cmd
		}
		if quake.Horizontal(s.Self, ride.exit) > 16 {
			return elevatorMove(cmd, s, ride.exit)
		}
		if math.Abs(s.Self[2]-ride.exit[2]) > 64 || p.Nav.AreaFor(s.Self) != ride.toArea {
			return cmd
		}
		p.routeIndex += 2
		p.elevator = nil
		p.World.Elevator = "completed"
		p.World.Route = p.route[p.routeIndex:]
		log.Printf("elevator model=%d completed map=%s frame=%d", ride.model, s.Map, s.Frame)
	}
	return cmd
}

func elevatorMove(cmd quake.UserCmd, s quake.Snapshot, target quake.Vec3) quake.UserCmd {
	cmd.Yaw = quake.YawTo(s.Self, target, s.DeltaAngles[1])
	cmd.Forward = 300
	return cmd
}
