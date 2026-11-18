package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class FeatureInteractionTest {

    private static String line(String title, Map<String, double[]> r) {
        StringBuilder b = new StringBuilder(title);
        r.forEach((k, v) -> {
            b.append(' ').append(k);
            for (double x : v) {
                b.append(String.format(" %.3f", x));
            }
            b.append(" |");
        });
        return b.toString();
    }

    @Test
    void interactionNetworksBeatLogisticOnThirdOrderCrosses() {
        Map<String, double[]> r = ClickModels.interactionOrder(3, 300, 1);
        String line = line("order-3 clicks AUC/logloss/params:", r);
        System.out.printf("%s%n", line);
        double logistic = r.get("logistic")[0];
        assertTrue(r.get("AutoInt")[0] > logistic, line);
        assertTrue(r.get("xDeepFM (CIN)")[0] > logistic, line);
        assertTrue(r.get("RankMixer")[0] > 0.55, line);
    }

    @Test
    void rankMixerTrainsDenseAndSparse() {
        Map<String, double[]> r = ClickModels.rankMixerSweep(new int[] {16}, 150, 1);
        String line = line("RankMixer params/AUC/ms-per-batch/active-experts:", r);
        System.out.printf("%s%n", line);
        assertTrue(r.get("RankMixer d=16")[1] > 0.55, line);
        assertTrue(r.get("RankMixer-MoE d=16")[1] > 0.55, line);
        assertTrue(r.get("RankMixer-MoE d=16")[3] <= 4, line);
    }

    @Test
    void compactEmbeddingsKeepMostOfTheAuc() {
        Map<String, double[]> r = ClickModels.memorySweep(new double[] {0.1}, 200, 1);
        String line = line("compact embeddings bytes-ratio/AUC/tail-AUC:", r);
        System.out.printf("%s%n", line);
        double full = r.get("full")[1];
        assertTrue(r.get("QUOTIENT_REMAINDER @0.1")[0] < 0.2, line);
        assertTrue(r.get("QUOTIENT_REMAINDER @0.1")[1] > full - 0.1, line);
        assertTrue(r.get("HASHED @0.1")[1] > 0.55, line);
    }
}
