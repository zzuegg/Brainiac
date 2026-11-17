package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class NormalizationShootoutTest {

    @Test
    void deepStacksTrainWithEveryStableScheme() {
        Map<String, float[]> results = new LinkedHashMap<>();
        for (Map.Entry<String, TunableTransformerBlock.Spec> e : NormalizationShootout.variants().entrySet()) {
            results.put(e.getKey(), NormalizationShootout.run(e.getValue(), 60, 3e-3f, 1));
        }
        StringBuilder sb = new StringBuilder();
        results.forEach((k, v) -> sb.append(k).append(": ").append(v[0]).append(" -> ").append(v[1]).append('\n'));
        System.out.println(sb);
        for (String name : new String[] {"pre-LN", "peri-LN", "pre-LN + QK-norm", "DyT", "ReZero", "LayerScale"}) {
            float[] v = results.get(name);
            assertTrue(v[1] < 0.6f * v[0], name + " learns:\n" + sb);
        }
    }
}
