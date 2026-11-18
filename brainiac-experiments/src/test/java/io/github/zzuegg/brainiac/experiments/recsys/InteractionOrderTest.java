package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class InteractionOrderTest {

    static String line(String title, Map<String, double[]> r) {
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
        Map<String, double[]> r = ClickModels.interactionOrder(3, 160, 1);
        String line = line("order-3 clicks AUC/logloss/params:", r);
        System.out.printf("%s%n", line);
        double logistic = r.get("logistic")[0];
        assertTrue(r.get("AutoInt")[0] > logistic, line);
        assertTrue(r.get("xDeepFM (CIN)")[0] > logistic, line);
        assertTrue(r.get("RankMixer")[0] > logistic, line);
    }
}
