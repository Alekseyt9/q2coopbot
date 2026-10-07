package trainingepisodes

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T) (string,Episode) {
	t.Helper(); r,err:=Load(filepath.Join("..","..","scripts","scenarios","combat-training","index.json"));if err!=nil{t.Fatal(err)}
	d:=t.TempDir();ep:=r.Episodes[0]
	write(t,filepath.Join(d,"index.json"),Manifest{Version:1,Files:[]string{"scene.json"}})
	write(t,filepath.Join(d,"scene.json"),ep)
	return d,ep
}
func write(t *testing.T,path string,v any){t.Helper();b,err:=json.Marshal(v);if err!=nil{t.Fatal(err)};if err=os.WriteFile(path,b,0600);err!=nil{t.Fatal(err)}}
func TestRegistryRejectsLeakageAndUnregisteredScenes(t *testing.T){
	d,ep:=fixture(t);ep.Splits["test"]=ep.Splits["train"];write(t,filepath.Join(d,"scene.json"),ep)
	if _,err:=Load(filepath.Join(d,"index.json"));err==nil||!strings.Contains(err.Error(),"leakage"){t.Fatalf("leaked seeds accepted: %v",err)}
	d,ep=fixture(t);write(t,filepath.Join(d,"extra.json"),ep)
	if _,err:=Load(filepath.Join(d,"index.json"));err==nil||!strings.Contains(err.Error(),"unregistered"){t.Fatalf("unlisted scene accepted: %v",err)}
}
func TestPlanRejectsPlannedSceneAndInvalidSplitBudget(t *testing.T){
	root,err:=filepath.Abs(filepath.Join("..",".."));if err!=nil{t.Fatal(err)}
	r,err:=Load(filepath.Join(root,"scripts","scenarios","combat-training","index.json"));if err!=nil{t.Fatal(err)}
	for _,tc:=range []struct{id,split string;count,offset int}{
		{"soldier-blaster-solo","train",4,0},
		{"parasite-blaster-solo","test",4,63},
		{"parasite-blaster-solo","validation",36,0},
		{"base1-natural-campaign","train",4,0},
		{"missing","train",4,0},
	}{if _,err:=Build(r,root,[]string{tc.id},tc.split,"rules","",t.TempDir(),tc.count,tc.offset);err==nil{t.Fatalf("invalid plan accepted: %+v",tc)}}
	p,err:=Build(r,root,[]string{"parasite-blaster-solo","parasite-gunner-blaster"},"train","rules","",t.TempDir(),4,8);if err!=nil{t.Fatal(err)}
	if len(p.Tasks)!=2||p.Workers!=4||p.Tasks[0].Seeds[0]!=100008||p.Tasks[1].Seeds[0]!=110008{t.Fatalf("wrong scene/seed schedule: %+v",p)}
}
func TestRegistryRejectsTraversalAndUnknownParameters(t *testing.T){
	d,_:=fixture(t);write(t,filepath.Join(d,"index.json"),Manifest{Version:1,Files:[]string{"../scene.json"}})
	if _,err:=Load(filepath.Join(d,"index.json"));err==nil{t.Fatal("path traversal accepted")}
	d,_=fixture(t);b,err:=os.ReadFile(filepath.Join(d,"scene.json"));if err!=nil{t.Fatal(err)}
	b=append([]byte(`{"shell_command":"bad",`),b[1:]...);if err=os.WriteFile(filepath.Join(d,"scene.json"),b,0600);err!=nil{t.Fatal(err)}
	if _,err:=Load(filepath.Join(d,"index.json"));err==nil{t.Fatal("unknown executable field accepted")}
}
