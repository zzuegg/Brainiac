package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.QuartetDense.Precision;

import org.junit.jupiter.api.Test;

class Fp4TrainingTest {

    @Test
    void quartetClosesMostOfTheFp4Gap() {
        double[] loss = new double[Precision.values().length];
        for (Precision p : Precision.values()) {
            for (long seed = 1; seed <= 2; seed++) {
                loss[p.ordinal()] += Fp4Training.train(p, 300, seed) / 2;
            }
        }
        double fp32 = loss[Precision.FP32.ordinal()];
        double quartet = loss[Precision.QUARTET.ordinal()];
        double naive = loss[Precision.FP4_NAIVE.ordinal()];
        double fp8 = loss[Precision.FP8.ordinal()];
        System.out.printf("FP4 training (loss after 300 steps): Quartet %.3f vs naive FP4 %.3f, FP8 %.3f, FP32 %.3f; "
                + "gap to FP32 %+.3f vs %+.3f / %+.3f%n", quartet, naive, fp8, fp32, quartet - fp32, naive - fp32,
                fp8 - fp32);
        assertTrue(quartet - fp32 < 0.7 * (naive - fp32), "Quartet closes much of naive FP4's gap");
        assertTrue(quartet < 1.15 * fp32, "and close to float32");
    }
}
