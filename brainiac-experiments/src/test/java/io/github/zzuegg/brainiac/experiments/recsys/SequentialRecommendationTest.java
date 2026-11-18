package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SequentialRecommendationTest {

    @Test
    void transformersUseOrderThatBprIgnores() {
        SequentialRecommendation task = new SequentialRecommendation(250, 120, 22, 8, 1);
        Map<String, SequentialRecommendation.Score> r = task.nextItem(200);
        StringBuilder line = new StringBuilder("next item HR@10/NDCG@10:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f", k, v.hr(), v.ndcg())));
        System.out.println(line);
        assertTrue(r.get("SASRec").hr() > r.get("BPR-MF").hr() + 0.1, line.toString());
        assertTrue(r.get("SASRec").hr() > r.get("popularity").hr() + 0.2, line.toString());
        assertTrue(r.get("BERT4Rec").hr() > r.get("popularity").hr() + 0.1, line.toString());
    }
}
