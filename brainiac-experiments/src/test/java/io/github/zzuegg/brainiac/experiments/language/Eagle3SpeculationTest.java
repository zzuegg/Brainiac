package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class Eagle3SpeculationTest {

    @Test
    void eagle3DraftsLosslesslyAndParallelDraftingCutsDraftPasses() {
        Map<String, Eagle3Speculation.Outcome> r = Eagle3Speculation.run(24, 100, 4, 20, 1);
        StringBuilder line = new StringBuilder("EAGLE-3, tokens per target pass / draft passes / tok/s:");
        r.forEach((k, v) -> line.append(String.format(" %s %.2f/%d/%.0f;", k, v.tokensPerPass(), v.draftPasses(),
                v.tokensPerSecond())));
        System.out.println(line);
        for (Map.Entry<String, Eagle3Speculation.Outcome> e : r.entrySet()) {
            assertArrayEquals(r.get("greedy").tokens(), e.getValue().tokens(), e.getKey());
        }
        // at toy scale EAGLE-3 trails EAGLE (see the class doc); its recipe must still draft usefully
        assertTrue(r.get("eagle-3 chain").tokensPerPass() > 1.5, line.toString());
        Eagle3Speculation.Outcome parallel = r.get("p-eagle (parallel chain)");
        assertTrue(parallel.tokensPerPass() > 0.9 * r.get("eagle-3 chain").tokensPerPass(), line.toString());
        assertTrue(parallel.draftPasses() * 2 < r.get("eagle-3 chain").draftPasses(), line.toString());
    }
}
