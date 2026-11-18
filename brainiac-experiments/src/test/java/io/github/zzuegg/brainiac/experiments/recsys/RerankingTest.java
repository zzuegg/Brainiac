package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class RerankingTest {

    @Test
    void listwiseAndLambdaLossesRerankTheHybrid() {
        Map<String, double[]> r = new RetrievalShootout(1).reranking(300, 60);
        StringBuilder line = new StringBuilder("rerank NDCG@5/NDCG@10:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f", k, v[0], v[1])));
        System.out.println(line);
        double best = Math.max(r.get("LambdaMART")[1], Math.max(r.get("LambdaRank")[1], r.get("ListNet")[1]));
        assertTrue(best > r.get("hybrid order")[1], line.toString());
        assertTrue(r.get("RankNet")[1] > 0.5, line.toString());
    }
}
