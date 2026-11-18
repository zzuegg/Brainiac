package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class GenerativeRetrievalTest {

    @Test
    void semanticIdsGenerateNextItemsIncludingColdOnes() {
        Map<String, GenerativeRetrieval.Score> r = new GenerativeRetrieval(250, 150, 16, 4, 1).run(170);
        StringBuilder line = new StringBuilder("generative retrieval recall@10 all/cold (collisions):");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f (%.2f)", k, v.recall(), v.coldRecall(),
                v.collisions())));
        System.out.println(line);
        GenerativeRetrieval.Score tiger = r.get("TIGER depth 2");
        assertTrue(tiger.recall() > 0.3, line.toString());
        assertTrue(tiger.coldRecall() > r.get("SASRec").coldRecall(), line.toString());
    }
}
