package io.github.zzuegg.brainiac.studio.builder;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Layers found from the core's API, and graphs checked and built with the core's own shape rules. */
class ModelGraphTest {

    private static LayerCatalog catalog;
    private static ModelGraph graphs;

    private static final String CONV = "io.github.zzuegg.brainiac.nn.Conv#of(int,int[])";
    private static final String POOL = "io.github.zzuegg.brainiac.nn.GlobalPool#avg()";
    private static final String DENSE = "io.github.zzuegg.brainiac.nn.Dense#spec(int,Activation,Init)";

    @BeforeAll
    static void find() {
        catalog = LayerCatalog.discover();
        graphs = new ModelGraph(catalog);
    }

    private static Map<String, Object> graph(String json) {
        return Json.object(Json.parse(json));
    }

    @Test
    void theCoreLayersAndTheirOptionsAreFound() {
        LayerCatalog.Entry conv = catalog.entry(CONV).orElseThrow(() -> new AssertionError(catalog.entries().stream()
                .map(LayerCatalog.Entry::key).filter(k -> k.contains("Conv")).toList()));
        assertEquals("Conv", conv.group());
        assertEquals(List.of("filters", "kernel"), conv.params().stream().map(LayerCatalog.Param::name).toList());
        assertTrue(conv.params().get(1).array());
        List<String> options = conv.options().stream().map(LayerCatalog.Option::key).toList();
        assertTrue(options.containsAll(List.of("stride(int[])", "padding(Padding)", "padding(int[])", "activation(Activation)",
                "bias(boolean)", "groups(int)")), options.toString());
        LayerCatalog.Param padding = conv.option("padding(Padding)").orElseThrow().params().getFirst();
        assertEquals(LayerCatalog.Kind.CHOICE, padding.kind());
        assertTrue(padding.choices().containsAll(List.of("SAME", "VALID", "FULL")), padding.choices().toString());
        LayerCatalog.Param act = catalog.entry(DENSE).orElseThrow().params().get(1);
        assertTrue(act.choices().containsAll(List.of("RELU", "GELU", "SOFTMAX", "IDENTITY")), act.choices().toString());
        assertTrue(catalog.entry(POOL).isPresent());
        assertTrue(catalog.entries().stream().anyMatch(e -> e.group().equals("MultiHeadAttention")));
        assertTrue(catalog.entries().stream().anyMatch(e -> e.group().equals("Recurrent") && e.name().equals("lstm")));
        // a factory that needs a tensor of weights cannot be filled in from the builder
        assertTrue(catalog.skipped().stream().anyMatch(k -> k.contains("Embedding#pretrained")), catalog.skipped().toString());
        Map<String, Object> json = catalog.toJson();
        assertFalse(Json.array(json.get("entries")).isEmpty());
        Json.write(json);
    }

    @Test
    void aResidualNetworkChecksOutAndBuilds() {
        Map<String, Object> g = graph("""
                {"nodes": [
                  {"id": "in", "op": "input", "shape": [3, 16, 16]},
                  {"id": "c1", "op": "layer", "entry": "%1$s", "args": [8, [3, 3]],
                   "options": [{"key": "padding(Padding)", "args": ["SAME"]}, {"key": "activation(Activation)", "args": ["RELU"]}],
                   "inputs": ["in"]},
                  {"id": "c2", "op": "layer", "entry": "%1$s", "args": [8, [3, 3]],
                   "options": [{"key": "padding(Padding)", "args": ["SAME"]}], "inputs": ["c1"]},
                  {"id": "sum", "op": "add", "inputs": ["c1", "c2"]},
                  {"id": "gap", "op": "layer", "entry": "%2$s", "args": [], "inputs": ["sum"]},
                  {"id": "out", "op": "layer", "entry": "%3$s", "args": [10, "SOFTMAX", "XAVIER"], "inputs": ["gap"]}
                ]}""".formatted(CONV, POOL, DENSE));
        ModelGraph.Check c = graphs.check(g);
        assertTrue(c.ok(), c.toJson().toString());
        assertEquals(Shape.of(8, 16, 16), c.shapes().get("c2"));
        assertEquals("in", c.input());
        assertEquals("out", c.output());
        assertEquals(List.of(10), c.toJson().get("outputShape"));
        Model m = graphs.build(g, 7);
        assertEquals(Shape.of(10), m.outputShape());
        // conv 3*8*9+8, conv 8*8*9+8, dense 8*10+10
        assertEquals(224 + 584 + 90, m.parameterCount());
    }

    @Test
    void aBlockThatCannotTakeItsInputSaysWhyAndBlocksWhatFollows() {
        Map<String, Object> g = graph("""
                {"nodes": [
                  {"id": "in", "op": "input", "shape": [3, 16, 16]},
                  {"id": "c1", "op": "layer", "entry": "%1$s", "args": [8, [3, 3]],
                   "options": [{"key": "stride(int[])", "args": [[2, 2]]}], "inputs": ["in"]},
                  {"id": "c2", "op": "layer", "entry": "%1$s", "args": [8, [3, 3]], "inputs": ["in"]},
                  {"id": "sum", "op": "add", "inputs": ["c1", "c2"]},
                  {"id": "out", "op": "layer", "entry": "%2$s", "args": [], "inputs": ["sum"]}
                ]}""".formatted(CONV, POOL));
        ModelGraph.Check c = graphs.check(g);
        assertFalse(c.ok());
        assertTrue(c.errors().get("sum").contains("cannot add shapes"), c.errors().toString());
        assertTrue(c.errors().get("out").contains("waits for sum"), c.errors().toString());
        assertThrows(IllegalArgumentException.class, () -> graphs.build(g, 1));
    }

    @Test
    void badValuesAndLoopsAreRefusedBlockByBlock() {
        Map<String, Object> g = graph("""
                {"nodes": [
                  {"id": "in", "op": "input", "shape": [4]},
                  {"id": "d1", "op": "layer", "entry": "%1$s", "args": [2.5, "RELU", "HE"], "inputs": ["in"]},
                  {"id": "d2", "op": "layer", "entry": "%1$s", "args": [4, "NOPE", "HE"], "inputs": ["in"]},
                  {"id": "d3", "op": "layer", "entry": "no.such#layer()", "inputs": ["in"]},
                  {"id": "a", "op": "add", "inputs": ["b", "in"]},
                  {"id": "b", "op": "add", "inputs": ["a", "in"]},
                  {"id": "lonely", "op": "layer", "entry": "%1$s", "args": [4, "RELU", "HE"]}
                ]}""".formatted(DENSE));
        ModelGraph.Check c = graphs.check(g);
        assertTrue(c.errors().get("d1").contains("whole number"), c.errors().toString());
        assertTrue(c.errors().get("d2").contains("has no NOPE"), c.errors().toString());
        assertTrue(c.errors().get("d3").contains("no layer"), c.errors().toString());
        assertTrue(c.errors().get("a").contains("loop"), c.errors().toString());
        assertTrue(c.errors().get("lonely").contains("not connected"), c.errors().toString());
        assertThrows(IllegalArgumentException.class, () -> graphs.check(graph("{\"nodes\": [{\"id\": \"x\"}, {\"id\": \"x\"}]}")));
    }
}
