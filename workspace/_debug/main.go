package main
import("fmt";"q2coopbot/internal/quake")
func main(){g,e:=quake.LoadMap("workspace/runtime/q2go/baseq2","base3");if e!=nil{panic(e)};for _,o:=range g.Entities{if o.Class=="info_player_start"||o.Class=="info_player_coop"{fmt.Printf("%+v\n",o)}}}
