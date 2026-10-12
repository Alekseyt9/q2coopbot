using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Collections.Generic;

// Parse the complete JSON first; project only supervisor/capture-proof fields.
// This representation must never replace the original training trace.
public static class CombatSupervisionProjection20261012 {
    static readonly HashSet<string> Root = new HashSet<string>(StringComparer.OrdinalIgnoreCase){
        "health","map","connection","spawncount","observation_frame","self_entity","sent_command"};
    static readonly HashSet<string> Control = new HashSet<string>(StringComparer.OrdinalIgnoreCase){
        "provider","applied_command","command_at_unix_ns"};
    static readonly HashSet<string> Selection = new HashSet<string>(StringComparer.OrdinalIgnoreCase){
        "owner","mode","candidate_command","interventions","elapsed_us"};
    static readonly HashSet<string> Observation = new HashSet<string>(StringComparer.OrdinalIgnoreCase){"version","identity","health"};
    static readonly HashSet<string> Applied = new HashSet<string>(StringComparer.OrdinalIgnoreCase){"version"};
    static bool Name(string a,string b) {return String.Equals(a,b,StringComparison.OrdinalIgnoreCase);}
    static void ValidateProperties(JsonElement element) {
        if(element.ValueKind==JsonValueKind.Array) {
            foreach(var item in element.EnumerateArray()) ValidateProperties(item);
        } else if(element.ValueKind==JsonValueKind.Object) {
            var names=new Dictionary<string,string>(StringComparer.OrdinalIgnoreCase);
            foreach(var p in element.EnumerateObject()) {
                if(p.Name.Length==0) throw new InvalidDataException("Empty JSON property name cannot be a PowerShell object field");
                string previous;
                if(names.TryGetValue(p.Name,out previous) && previous!=p.Name)
                    throw new InvalidDataException("JSON property names differ only by case");
                names[p.Name]=p.Name;
                ValidateProperties(p.Value);
            }
        }
    }
    static void Fields(Utf8JsonWriter writer, JsonElement element, HashSet<string> selected) {
        if(element.ValueKind != JsonValueKind.Object) {element.WriteTo(writer);return;}
        writer.WriteStartObject();
        foreach(var p in element.EnumerateObject()) {
            if(selected.Contains(p.Name)) p.WriteTo(writer);
        }
        writer.WriteEndObject();
    }
    public static string Project(string line) {
        using(var document = JsonDocument.Parse(line,new JsonDocumentOptions{MaxDepth=1024}))
        using(var stream = new MemoryStream()) {
            var root=document.RootElement;
            if(root.ValueKind!=JsonValueKind.Object) throw new InvalidDataException("Trace row must be an object");
            ValidateProperties(root);
            using(var writer=new Utf8JsonWriter(stream)) {
                writer.WriteStartObject();
                foreach(var p in root.EnumerateObject()) {
                    if(Root.Contains(p.Name)) {p.WriteTo(writer);continue;}
                    if(!Name(p.Name,"combat_policy")) continue;
                    writer.WritePropertyName(p.Name);
                    if(p.Value.ValueKind!=JsonValueKind.Object) {p.Value.WriteTo(writer);continue;}
                    writer.WriteStartObject();
                    foreach(var c in p.Value.EnumerateObject()) {
                        if(Control.Contains(c.Name)) {c.WriteTo(writer);continue;}
                        if(Name(c.Name,"selection")) {writer.WritePropertyName(c.Name);Fields(writer,c.Value,Selection);}
                        else if(Name(c.Name,"observation")) {writer.WritePropertyName(c.Name);Fields(writer,c.Value,Observation);}
                        else if(Name(c.Name,"applied")) {writer.WritePropertyName(c.Name);Fields(writer,c.Value,Applied);}
                    }
                    writer.WriteEndObject();
                }
                writer.WriteEndObject();
            }
            return Encoding.UTF8.GetString(stream.ToArray());
        }
    }
    public static IEnumerable<string> Read(string path) {
        foreach(var line in File.ReadLines(path,new UTF8Encoding(false,true))) {
            if(!string.IsNullOrWhiteSpace(line)) yield return Project(line);
        }
    }
}
