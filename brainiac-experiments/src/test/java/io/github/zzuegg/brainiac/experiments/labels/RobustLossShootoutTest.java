package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Noise;
import io.github.zzuegg.brainiac.experiments.labels.NoisyLabelShootout.Result;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class RobustLossShootoutTest {

    @Test
    void robustLossesUnderFortyPercentSymmetricNoise() {
        Map<Method, Result> rows = new EnumMap<>(Method.class);
        for (Method m : new Method[] {Method.CE, Method.LABEL_SMOOTHING, Method.FOCAL, Method.MAE, Method.GCE,
                Method.SCE, Method.NCE_RCE, Method.JAL}) {
            Result r = NoisyLabelShootout.run(m, Noise.SYMMETRIC, 0.4, 600, 1);
            rows.put(m, r);
            print("sym 0.4", r);
        }
        double bestRobust = 0;
        for (Method m : new Method[] {Method.GCE, Method.SCE, Method.NCE_RCE, Method.JAL}) {
            bestRobust = Math.max(bestRobust, rows.get(m).finalAccuracy());
            assertTrue(rows.get(Method.CE).noisyFit() > rows.get(m).noisyFit() + 0.2, m + " memorises less than CE");
        }
        assertTrue(bestRobust > rows.get(Method.CE).finalAccuracy(), "the best robust loss beats CE");
        for (Result r : rows.values()) {
            assertTrue(r.bestAccuracy() > 0.2, "every loss trains: " + r);
        }
    }

    @Test
    void pairFlipNoise() {
        for (Method m : new Method[] {Method.CE, Method.NCE_RCE, Method.JAL}) {
            Result r = NoisyLabelShootout.run(m, Noise.PAIR_FLIP, 0.4, 600, 1);
            print("pair 0.4", r);
            assertTrue(r.bestAccuracy() > 0.2, r.toString());
        }
    }

    static void print(String setting, Result r) {
        System.out.printf("Noisy %-8s %-15s final %.3f best %.3f  flipped fit %.2f corrected %.2f  split P/R %.2f/%.2f  T err %.3f%n",
                setting, r.method(), r.finalAccuracy(), r.bestAccuracy(), r.noisyFit(), r.noisyCorrected(),
                r.splitPrecision(), r.splitRecall(), r.transitionError());
    }
}
