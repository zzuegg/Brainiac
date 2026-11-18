package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class LocalForwardGradientComparisonTest {

    @Test
    void localActivityForwardGradientsCloseMostOfTheGapToBackprop() {
        Map<String, LocalForwardGradientComparison.Row> r = LocalForwardGradientComparison.spirals(30, 1);
        StringBuilder s = new StringBuilder("Spirals 2-32-32-3, 30 epochs, test accuracy/gradient cosine/ms:");
        r.forEach((k, v) -> s.append(String.format(" %s %.3f/%.2f/%d;", k, v.result().testAccuracy(), v.cosine(),
                v.result().millis())));
        System.out.println(s);
        double weight = r.get("forward gradient (weights, 32 dirs)").result().testAccuracy();
        LocalForwardGradientComparison.Row local = r.get("local activity FG (4 groups x 2 dirs)");
        assertTrue(local.result().testAccuracy() > weight, s.toString());
        assertTrue(local.cosine() > r.get("forward gradient (weights, 32 dirs)").cosine(), s.toString());
    }
}
