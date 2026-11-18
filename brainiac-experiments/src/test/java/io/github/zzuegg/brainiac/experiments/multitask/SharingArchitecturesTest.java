package io.github.zzuegg.brainiac.experiments.multitask;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;

class SharingArchitecturesTest {

    @Test
    void softSharing() {
        for (String name : new String[] {"related", "conflicting"}) {
            Map<String, double[]> r = SharingArchitectures.run(name, 32, 400, 1);
            r.forEach((k, v) -> System.out.printf("sharing %s %s: delta_MTL %+.1f%% params %.0f, rest %s%n", name, k,
                    v[0], v[1], Arrays.toString(Arrays.stream(v, 2, v.length).map(x -> Math.round(x * 1000) / 1000.0)
                            .toArray())));
            assertTrue(Double.isFinite(r.get("cross-stitch")[0]), "cross-stitch trains");
        }
    }
}
