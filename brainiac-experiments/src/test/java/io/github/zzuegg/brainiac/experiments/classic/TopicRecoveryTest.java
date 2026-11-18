package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TopicRecoveryTest {

    @Test
    void ldaRecoversPlantedTopicsBetterThanKMeans() {
        TopicRecovery.Corpus corpus = TopicRecovery.corpus(6, 60, 300, 60, 0.5, 1);
        Map<String, double[]> r = TopicRecovery.compare(corpus, 150, 2);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s cos %.3f ppl %.2f; ", k, v[0], v[1])));
        System.out.printf("planted topics: %s%n", sb);
        assertTrue(r.get("LDA (Gibbs)")[0] > 0.9, sb.toString());
        assertTrue(r.get("LDA (Gibbs)")[0] > r.get("k-means on BoW")[0], sb.toString());
        assertTrue(r.get("LDA (Gibbs)")[1] < r.get("k-means on BoW")[1], sb.toString());
        assertTrue(r.get("NMF")[0] > 0.8, sb.toString());
        assertTrue(r.get("LDA (Gibbs)")[1] < r.get("true topics")[1] * 1.1, sb.toString());
    }
}
