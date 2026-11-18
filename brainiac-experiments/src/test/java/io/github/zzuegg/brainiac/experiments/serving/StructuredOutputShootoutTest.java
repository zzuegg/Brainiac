package io.github.zzuegg.brainiac.experiments.serving;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class StructuredOutputShootoutTest {

    @Test
    void maskCacheGivesIdenticalOutputsFaster() {
        Map<String, Map<String, StructuredOutputShootout.Row>> r = StructuredOutputShootout.run(40, 40, 1);
        r.forEach((case_, rows) -> rows.forEach((k, v) -> System.out.printf(
                "8.4.9 %-20s %-32s %.1f us/token, hit rate %.2f, %d states, identical %s%n", case_, k,
                v.microsPerToken(), v.hitRate(), v.states(), v.identical())));
        for (Map<String, StructuredOutputShootout.Row> rows : r.values()) {
            rows.values().forEach(v -> assertTrue(v.identical(), rows.toString()));
        }
        Map<String, StructuredOutputShootout.Row> json = r.get("JSON / BPE 512");
        assertTrue(json.get("GrammarMaskCache (precomputed)").microsPerToken()
                < json.get("per-token scan").microsPerToken() / 3, json.toString());
    }
}
