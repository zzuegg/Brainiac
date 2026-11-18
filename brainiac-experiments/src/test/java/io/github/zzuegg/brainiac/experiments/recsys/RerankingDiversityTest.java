package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class RerankingDiversityTest {

    private static String line(String title, Map<String, double[]> r) {
        StringBuilder b = new StringBuilder(title);
        r.forEach((k, v) -> b.append(String.format(" %s %.3f/%.3f/%.3f |", k, v[0], v[1], v[2])));
        return b.toString();
    }

    @Test
    void diverseRerankersCoverMoreTopics() {
        Map<String, double[]> r = RerankingDiversity.run(false, 0, 1);
        String line = line("rerank NDCG@10/alpha-NDCG@10/ILD (noisy oracle):", r);
        System.out.println(line);
        double[] base = r.get("relevance only");
        assertTrue(r.get("DPP theta 0.7")[2] > base[2], line);
        assertTrue(r.get("MMR lambda 0.5")[2] > base[2], line);
        assertTrue(Math.max(r.get("DPP theta 0.7")[1], r.get("DPP theta 0.9")[1]) > base[1], line);
    }

    @Test
    void twoTowerCandidatesCanBeDiversified() {
        Map<String, double[]> r = RerankingDiversity.run(true, 300, 1);
        String line = line("rerank NDCG@10/alpha-NDCG@10/ILD (two-tower):", r);
        System.out.println(line);
        assertTrue(r.get("DPP theta 0.7")[2] > r.get("relevance only")[2], line);
    }
}
