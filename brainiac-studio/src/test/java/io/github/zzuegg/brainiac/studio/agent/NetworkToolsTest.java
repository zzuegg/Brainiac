package io.github.zzuegg.brainiac.studio.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.builder.Graphs;
import io.github.zzuegg.brainiac.studio.items.Owners;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The builder's agent tools change a network a few blocks at a time and never break one that works. */
class NetworkToolsTest {

    private static final String DENSE = "io.github.zzuegg.brainiac.nn.Dense#spec(int,Activation,Init)";

    @TempDir
    Path folder;

    private static Map<String, Object> step(String json) {
        return Json.object(Json.parse(json));
    }

    @Test
    void anAgentGrowsANetworkAndCannotBreakIt() throws Exception {
        Graphs graphs = new Graphs(folder);
        NetworkTools tools = new NetworkTools(graphs, new Owners(folder.resolve("owners.json")));
        tools.createNetwork("tiny", List.of(8));
        assertEquals(1, tools.listNetworks().size());

        Map<String, Object> grown = tools.editNetwork("tiny", List.of(
                step("{\"do\": \"add\", \"id\": \"h\", \"layer\": \"" + DENSE + "\", \"args\": {\"units\": 16}, \"inputs\": [\"in\"]}"),
                step("{\"do\": \"add\", \"layer\": \"" + DENSE + "\", \"inputs\": [\"h\"]}"),
                step("{\"do\": \"output\", \"id\": \"dense1\"}")));
        assertEquals(true, grown.get("ok"), String.valueOf(grown));
        assertEquals(List.of("h", "dense1"), grown.get("added"));
        Map<String, Object> h = Json.object(Json.array(grown.get("blocks")).get(1));
        assertEquals(16, ((Number) Json.array(h.get("args")).getFirst()).intValue());

        // removing the input would break both dense blocks: refused, and nothing is saved
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> tools.editNetwork("tiny", List.of(step("{\"do\": \"remove\", \"id\": \"in\"}"))));
        assertTrue(e.getMessage().contains("refused"), e.getMessage());
        assertEquals(3, Json.array(graphs.get("tiny").orElseThrow().get("nodes")).size());

        // a bad step names itself and changes nothing either
        e = assertThrows(IllegalArgumentException.class, () -> tools.editNetwork("tiny", List.of(
                step("{\"do\": \"remove\", \"id\": \"dense1\"}"), step("{\"do\": \"connect\", \"from\": \"nope\", \"to\": \"h\"}"))));
        assertTrue(e.getMessage().startsWith("step 2 (connect)"), e.getMessage());
        assertEquals(3, Json.array(tools.readNetwork("tiny").get("blocks")).size());
    }
}
