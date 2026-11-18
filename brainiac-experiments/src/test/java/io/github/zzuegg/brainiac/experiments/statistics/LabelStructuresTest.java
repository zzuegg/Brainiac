package io.github.zzuegg.brainiac.experiments.statistics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class LabelStructuresTest {

    private static String fmt(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append(' ');
            for (double x : v) {
                sb.append(String.format("%.3f ", x));
            }
            sb.append("; ");
        });
        return sb.toString();
    }

    @Test
    void ordinalHeadsCutMaeAndStayRankConsistent() {
        Map<String, double[]> r = LabelStructures.ordinal(600, 300, 1);
        System.out.printf("ordinal K=6 {MAE, acc, violations}: %s%n", fmt(r));
        assertTrue(Math.min(r.get("CORAL")[0], r.get("CORN")[0]) < r.get("softmax")[0], fmt(r));
        assertTrue(r.get("CORN")[2] == 0 && r.get("PROPORTIONAL_ODDS")[2] == 0, fmt(r));
    }

    @Test
    void chainsBeatBinaryRelevanceInSubsetAccuracy() {
        Map<String, double[]> r = LabelStructures.multiLabel(800, 2);
        System.out.printf("multi-label {Hamming, subset acc, micro-F1}: %s%n", fmt(r));
        assertTrue(r.get("classifier chain")[1] > r.get("binary relevance")[1], fmt(r));
        assertTrue(r.get("label powerset")[1] > r.get("binary relevance")[1], fmt(r));
    }

    @Test
    void resamplingRaisesEceAndRecalibrationRepairsIt() {
        Map<String, double[]> r = LabelStructures.imbalance(50, 3000, 3);
        System.out.printf("imbalance 1:50 {AUROC, minF1, bacc, ECE, ECE iso}: %s%n", fmt(r));
        assertTrue(r.get("RANDOM_UNDER")[3] > r.get("plain")[3], fmt(r));
        assertTrue(r.get("RANDOM_UNDER + prior shift")[3] < r.get("RANDOM_UNDER")[3], fmt(r));
        assertTrue(r.get("SMOTE")[4] < r.get("SMOTE")[3], fmt(r));
    }

    @Test
    void interactionAwareSelectorsFindXorPairs() {
        Map<String, double[]> r = LabelStructures.featureSelection(600, 30, 4);
        System.out.printf("feature selection {recall@5, recall@8, redundant@8, RF acc, ms}: %s%n", fmt(r));
        assertTrue(r.get("RELIEFF")[1] > r.get("lasso path (entry lambda)")[1], fmt(r));
        assertTrue(r.get("BORUTA")[1] >= 0.6, fmt(r));
    }
}
