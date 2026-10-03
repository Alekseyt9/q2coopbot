package main

import (
	"encoding/xml"
	"q2coopbot/internal/quake"
	"testing"
)

func TestSVGWorldCoordinates(t *testing.T) {
	data := outlineSVG(quake.MapOutline{Floors: [][]quake.Vec3{{{10, 20, 30}, {40, 20, 30}, {40, 50, 60}}}})
	var svg struct {
		ViewBox string `xml:"viewBox,attr"`
		Group   struct {
			Polygons []struct {
				Points string `xml:"points,attr"`
				Z      string `xml:"data-z,attr"`
				Effect string `xml:"vector-effect,attr"`
			} `xml:"polygon"`
		} `xml:"g"`
	}
	if e := xml.Unmarshal(data, &svg); e != nil {
		t.Fatal(e)
	}
	if svg.ViewBox != "10.000 -50.000 30.000 30.000" || len(svg.Group.Polygons) != 1 {
		t.Fatalf("unexpected SVG: %s", data)
	}
	poly := svg.Group.Polygons[0]
	if poly.Points != "10.000,-20.000 40.000,-20.000 40.000,-50.000 " || poly.Z != "30.000 30.000 60.000 " || poly.Effect != "non-scaling-stroke" {
		t.Fatalf("unexpected coordinates: %+v", poly)
	}
}
