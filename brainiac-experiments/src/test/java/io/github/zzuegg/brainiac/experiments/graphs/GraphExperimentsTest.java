package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class GraphExperimentsTest {

    @Test
    void messagePassingSurvivesFeatureNoise() {
        Map<String, Double> r = GnnBenchmarks.nodeClassification(new double[] {2.0}, 80, 3);
        System.out.println("node classification: " + r);
        assertTrue(r.get("GCN @ 2.0") > r.get("MLP @ 2.0") + 0.15, r.toString());
    }

    @Test
    void sumAggregationCountsStructure() {
        Map<String, Double> r = GnnBenchmarks.graphClassification(150, 1);
        System.out.println("graph classification: " + r);
        assertTrue(r.get("GIN + sum") > r.get("SAGE + mean"), r.toString());
    }

    @Test
    void graphAutoencoderPredictsLinks() {
        Map<String, Double> r = GnnBenchmarks.linkPrediction(150, 2);
        System.out.println("link prediction AUC: " + r);
        assertTrue(r.get("GAE") > 0.7 && r.get("GAE") > r.get("common neighbours"), r.toString());
    }

    @Test
    void recommendationAndTabularModelsBeatBaselines() {
        Map<String, Double> retrieval = RecsysTabularBenchmarks.retrieval(300, 1);
        System.out.println("retrieval recall@10: " + retrieval);
        assertTrue(retrieval.get("two-tower (new users)") > 3 * retrieval.get("random"), retrieval.toString());
        assertTrue(retrieval.get("BPR-MF") > 3 * retrieval.get("random"), retrieval.toString());
        Map<String, Double> clicks = RecsysTabularBenchmarks.clicks(500, 1);
        System.out.println("click accuracy: " + clicks);
        assertTrue(clicks.get("FM") > clicks.get("logistic"), clicks.toString());
    }
}
