package main

import (
 "fmt"
 "math"
 "strings"
 "q2coopbot/internal/quake"
)

// SVG uses world X and inverted Y, preserving per-vertex Z for floor slices.
// Only numeric geometry is emitted; there are no scripts or external resources.
func outlineSVG(outline quake.MapOutline) []byte {
 minX,minY,maxX,maxY:=math.Inf(1),math.Inf(1),math.Inf(-1),math.Inf(-1)
 for _,floor:=range outline.Floors{for _,p:=range floor{minX=math.Min(minX,p[0]);minY=math.Min(minY,-p[1]);maxX=math.Max(maxX,p[0]);maxY=math.Max(maxY,-p[1])}}
 if len(outline.Floors)==0{minX,minY,maxX,maxY=0,0,1,1}
 var out strings.Builder
 fmt.Fprintf(&out,`<svg xmlns="http://www.w3.org/2000/svg" viewBox="%.3f %.3f %.3f %.3f"><g fill="#283343" stroke="#526176" stroke-width="0.65">`,minX,minY,math.Max(1,maxX-minX),math.Max(1,maxY-minY))
 for _,floor:=range outline.Floors {
  out.WriteString(`<polygon vector-effect="non-scaling-stroke" points="`)
  for _,p:=range floor{fmt.Fprintf(&out,"%.3f,%.3f ",p[0],-p[1])}
  out.WriteString(`" data-z="`);for _,p:=range floor{fmt.Fprintf(&out,"%.3f ",p[2])};out.WriteString(`"/>`)
 }
 out.WriteString(`</g></svg>`);return []byte(out.String())
}
