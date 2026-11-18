package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.DataValuation.Keep;
import io.github.zzuegg.brainiac.experiments.data.DataValuation.Prepared;

import org.junit.jupiter.api.Test;

class PrototypePruningTest {

    @Test
    void pruningDirectionAcrossDatasetSizes() {
        double full = 0;
        for (int size : new int[] {500, 2000, 8000}) {
            Prepared p = DataValuation.prepare(size, 1);
            for (double keep : new double[] {0.3, 0.6, 0.9}) {
                double[] acc = new double[Keep.values().length];
                for (Keep k : Keep.values()) {
                    acc[k.ordinal()] = DataValuation.pruning(p, keep, k, 250, 1);
                }
                System.out.printf("Prototype pruning n %4d keep %.1f: random %.3f  easy %.3f  hard %.3f  "
                        + "EL2N-hard %.3f%n", size, keep, acc[0], acc[1], acc[2], acc[3]);
            }
            if (size == 8000) {
                full = DataValuation.pruning(p, 1.0, Keep.RANDOM, 250, 1);
            }
        }
        System.out.printf("Prototype pruning n 8000 keep 1.0: %.3f%n", full);
        assertTrue(full > 0.5, "the full set trains");
    }
}
