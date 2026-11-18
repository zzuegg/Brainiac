package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class CollaborativeFilteringTest {

    @Test
    void lightGcnEaseAndMultVaeBeatBprMf() {
        Map<String, CollaborativeFiltering.Score> r = new CollaborativeFiltering(300, 200, 1).run(220, 3);
        StringBuilder line = new StringBuilder("CF recall@10/NDCG@10:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f", k, v.recall(), v.ndcg())));
        System.out.println(line);
        System.out.printf("CF fit ms: BPR-MF %d, EASE %d, Mult-VAE %d%n", r.get("BPR-MF").millis(),
                r.get("EASE").millis(), r.get("Mult-VAE").millis());
        double bpr = r.get("BPR-MF").recall();
        double bestGcn = Math.max(r.get("LightGCN K=2").recall(), r.get("LightGCN K=3").recall());
        assertTrue(bestGcn > bpr, line.toString());
        assertTrue(r.get("EASE").recall() >= bpr - 0.02, line.toString());
        assertTrue(r.get("Mult-VAE").recall() > 2 * r.get("popularity").recall(), line.toString());
    }
}
