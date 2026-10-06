package policy

import (
	"math"
	"testing"
)

func TestEntityTokenFeatureOffsetsAndMasks(t *testing.T) {
	x := make([]float64, 810)
	for _, j := range []int{73, 170, 25, 223, 260} {
		x[j] = 1
	}
	x[466+15] = 1
	x[386+1] = .75
	x[786+1] = .5
	tokens, e := entityTokens(x)
	if e != nil || len(tokens) != 6 {
		t.Fatal(len(tokens), e)
	}
	if tokens[0][26] != .5 || tokens[1][12+1] != .75 || tokens[1][22+15] != 1 || tokens[1][63] != 1 || tokens[2][64] != 1 {
		t.Fatal("v4 offsets/roles differ")
	}
	if _, e = entityTokens(x[:800]); e == nil {
		t.Fatal("bad feature width accepted")
	}
}
func TestAttentionCacheWindow(t *testing.T) {
	width := 4
	c := AttentionCell{Query: gruMatrix(width, width), Key: gruMatrix(width, width), Value: gruMatrix(width, width), Output: gruMatrix(width, width), Residual: gruMatrix(1, width)}
	for i := 0; i < width; i++ {
		c.Value.Weight[i][i] = 1
		c.Output.Weight[i][i] = 1
	}
	c.Residual.Weight[0][0] = 1
	previous := []float64{}
	for position := 0; position < 6; position++ {
		y, next := attentionForward(c, []float64{0, 0, 0, 0}, previous, 2, 3, position)
		if len(next) != min(position+1, 2)*width || math.IsNaN(y[0]) {
			t.Fatal("cache bounds", position, next)
		}
		if position == 0 && y[0] != 0 {
			t.Fatal("position0 sine")
		}
		previous = next
	}
}
