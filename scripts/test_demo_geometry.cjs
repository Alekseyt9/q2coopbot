const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../cmd/q2runs/web/demo3d.js'), 'utf8'), context);
const Renderer = context.window.SketchRenderer;
// A real BSP edge may be split by another brush: the initial triangle has
// zero area even though the polygon itself is a valid floor or wall.
const floor = [[0,0,0],[1,0,0],[2,0,0],[2,2,0],[0,2,0]];
const normal = Renderer.polygonNormal(floor);
assert.ok(Math.abs(normal[2]-1)<1e-8);
assert.ok(Math.abs(Renderer.polygonNormal([...floor].reverse())[2]+1)<1e-8);
const wall = floor.map(([x,y,z])=>[x,z,y]);
assert.ok(Math.abs(Renderer.polygonNormal(wall)[1]+1)<1e-8);
assert.equal(Renderer.polygonNormal([[0,0,0],[1,0,0],[2,0,0]]),null);
const uploaded = [];
const renderer = Object.create(Renderer.prototype);
renderer.gl = {createBuffer:()=>({}),bindBuffer:()=>{},bufferData:(_,data)=>uploaded.push(data)};
renderer.mesh([floor,wall]);
for(const buffer of uploaded)for(let i=0;i<buffer.length;i+=6){
  assert.ok(Math.abs(Math.hypot(buffer[i+3],buffer[i+4],buffer[i+5])-1)<1e-6);
}
console.log('BSP polygon normals: collinear starts, winding, walls and vertex buffers passed');
