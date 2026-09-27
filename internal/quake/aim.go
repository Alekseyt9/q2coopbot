package quake

func (s Snapshot) EyePoint() Vec3 {
	p := s.Self
	if s.Ducked {
		p[2] -= 2
	} else {
		p[2] += 22
	}
	return p
}

// AimPoint keeps the usual upper-body aim inside the server's current bbox.
// Protocol 34 packs radius/down/up in solid; 31 denotes a BSP brush, not a bbox.
// Missing bounds retain the legacy aim, without claiming a known posture.
func (o Object) AimPoint() Vec3 {
	p := o.Origin
	z := 22.0
	if o.Solid != 0 && o.Solid != 31 && o.Solid&31 != 0 {
		bottom := -float64((o.Solid>>5)&31) * 8
		top := float64((o.Solid>>10)&63)*8 - 32
		if top > bottom {
			if top-bottom < 16 {
				z = (top + bottom) / 2
			} else {
				z = min(max(z, bottom+8), top-8)
			}
		}
	}
	p[2] += z
	return p
}
