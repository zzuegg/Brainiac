package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Map;

import org.junit.jupiter.api.Test;

class HybridAndCanonShootoutTest {

    @Test
    void canonLayersLetOneBlockDoAssociativeRecall() {
        Map<String, Double> acc = HybridAndCanonShootout.canon(
                new ArrayList<>(HybridAndCanonShootout.canonLineup(32).keySet()), 32, 250, 1);
        System.out.printf("one-block recall without bigram feature: block %.3f -> +canon %.3f, core block %.3f -> "
                + "canon + core %.3f%n", acc.get("block"), acc.get("block + canon"), acc.get("core block"),
                acc.get("canon + core block"));
        assertTrue(acc.get("block + canon") > acc.get("block") + 0.2);
        assertTrue(acc.get("canon + core block") > acc.get("core block") + 0.2);
    }
}
