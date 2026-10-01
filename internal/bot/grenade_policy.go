package bot

import (
	"math"
	"q2coopbot/internal/quake"
)

// Practical sampled policy, not a continuous safety certificate. An already
// started cycle is always released; loss of the target never extends the fuse.
type grenadeThrow struct {
	Map string
	Started int
	Pitch, Yaw int16 // World angles, independent of server delta angles.
	Target int
	Samples []GrenadeFlight
}

func (p *Planner) grenadeThrowPending(s quake.Snapshot) bool {
	t:=p.grenadeThrow
	return t!=nil && t.Map==s.Map && s.Frame>=t.Started && s.Frame-t.Started<20 && s.Health>0 && isHandGrenade(s.Weapon)
}

func (p *Planner) chooseGrenadeThrow(s quake.Snapshot) *grenadeThrow {
	if p.TestDisableHandGrenade || s.Health<45 || !s.OnGround || s.Ducked || s.Gravity<=0 || quake.Distance(s.SelfVelocity,quake.Vec3{})>20 || !p.World.Geometry.HasCollision() { return nil }
	if p.grenadeThrow!=nil && p.grenadeThrow.Map==s.Map && s.Frame>=p.grenadeThrow.Started && s.Frame-p.grenadeThrow.Started<100 { return nil }
	if s.Teammate==nil && s.LastTeammate!=nil { return nil }
	m:=p.shotTeammateMotion
	if s.Teammate!=nil && (!m.known || m.frame!=s.Frame || m.entity!=s.TeammateEntity || quake.Distance(m.velocity,quake.Vec3{})>100 || m.velocityChange>40) { return nil }
	if len(s.Obstacles)>len(s.Enemies) { return nil }
	bodies:=[]grenadeBody{}
	var target *quake.Object
	for i:=range s.Enemies {
		e:=&s.Enemies[i]
		if e.Solid==0 || e.Solid==31 { return nil }
		x,down,top:=float64(e.Solid&31)*8+8,float64((e.Solid>>5)&31)*8+8,float64((e.Solid>>10)&63)*8-32+8
		bodies=append(bodies,grenadeBody{e.Origin,quake.Vec3{-x,-x,-down},quake.Vec3{x,x,top}})
		d:=quake.Distance(s.Self,e.Origin)
		if e.ClearShot!=nil && *e.ClearShot && d>=256 && d<=650 && (target==nil || d<quake.Distance(s.Self,target.Origin)) { target=e }
	}
	if target==nil { return nil }
	if s.Teammate!=nil { bodies=append(bodies,grenadeBody{*s.Teammate,quake.Vec3{-24,-24,-32},quake.Vec3{24,24,40}}) }
	g:=p.World.Geometry
	trace:=func(a,b quake.Vec3) quake.PointTrace {
		tr:=g.TraceProjectile(a,b)
		for _,mover:=range s.Movers {
			model,ok:=g.Model(mover.Model); if !ok { return quake.PointTrace{} }
			if f,hit:=grenadeBodyHit(a,b,grenadeBody{mover.Origin,model.Min,model.Max}); hit && f<=tr.Fraction { return quake.PointTrace{} }
		}
		return tr
	}
	goal:=target.Origin
	if motion,ok:=p.enemyMotion[target.ID]; ok && motion.frame==s.Frame && motion.stable && motion.velocityKnown {
		for i:=range goal { goal[i]+=motion.velocity[i]*.5 }
	}
	yaw:=quake.YawTo(s.Self,goal,0)
	y:=float64(yaw)*2*math.Pi/65536
	bestScore:=math.Inf(1)
	var best *grenadeThrow
	for _,degrees:=range []float64{0,-10,-20,10,20} {
		angle:=degrees*math.Pi/180
		sp,cp:=math.Sincos(angle); sy,cy:=math.Sincos(y)
		fwd,right,up:=quake.Vec3{cp*cy,cp*sy,-sp},quake.Vec3{sy,-cy,0},quake.Vec3{sp*cy,sp*sy,cp}
		start:=s.Self; for i:=range start { start[i]+=8*fwd[i] }; start[2]+=14
		candidate:=&grenadeThrow{Map:s.Map,Started:s.Frame,Pitch:int16(degrees*65536/360),Yaw:yaw,Target:target.ID}
		valid,score:=true,0.0
		for _,uj:=range []float64{-10,0,10} { for _,rj:=range []float64{-10,0,10} {
			v:=quake.Vec3{}; for i:=range v { v[i]=413*fwd[i]+(200+uj)*up[i]+rj*right[i] }
			flight:=grenadeFlight(start,v,2.9,float64(s.Gravity),trace,bodies)
			flight.Risk="sampled_acceptable"
			if flight.Event!="fuse" && flight.Event!="damageable_contact" || quake.Distance(flight.End,s.Self)<=225+quake.Distance(s.SelfVelocity,quake.Vec3{})*flight.Seconds { valid=false }
			if s.Teammate!=nil && (quake.Distance(flight.End,*s.Teammate)<=245+quake.Distance(m.velocity,quake.Vec3{})*flight.Seconds || grenadeFriendContact(start,v,float64(s.Gravity),trace,*s.Teammate,m.velocity)>0) { valid=false }
			if hit:=grenadeTargetContact(start,v,float64(s.Gravity),trace,s.Enemies,p.grenadeTargetMotions(s)); hit!=nil && (quake.Distance(hit.Point,s.Self)<=225 || s.Teammate!=nil && quake.Distance(hit.Point,*s.Teammate)<=245+quake.Distance(m.velocity,quake.Vec3{})*hit.Seconds) { valid=false }
			d:=quake.Distance(flight.End,goal); if d>160 { valid=false }; score+=d
			candidate.Samples=append(candidate.Samples,flight)
		} }
		if valid && score<bestScore { bestScore,best=score,candidate }
	}
	return best
}
