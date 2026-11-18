package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SpikingConversionTest {

    @Test
    void qcfsConvertsAtFewTimeSteps() {
        int[] steps = {1, 2, 4, 8, 16, 32, 64, 128, 256};
        double qcfsMinusPlainAt4 = 0;
        double plainAt256 = 0;
        double ann = 0;
        for (int depth : new int[] {1, 3}) {
            SpikingClassifier.Conversion c = SpikingClassifier.conversion(steps, 8, depth, 400, 1);
            StringBuilder row = new StringBuilder();
            for (int i = 0; i < steps.length; i++) {
                row.append(String.format(" T%d %.3f/%.3f", steps[i], c.plain()[i], c.qcfs()[i]));
            }
            System.out.printf("SNN conversion digits, %d hidden: ReLU ANN %.3f, QCFS(L=8) ANN %.3f; plain/QCFS:%s%n",
                    depth, c.ann(), c.qcfsAnn(), row);
            System.out.printf("SNN conversion %d hidden: spikes per example at T 8: plain %.0f, QCFS %.0f%n", depth,
                    c.plainSpikes()[3], c.qcfsSpikes()[3]);
            if (depth == 3) {
                qcfsMinusPlainAt4 = c.qcfs()[2] - c.plain()[2];
                plainAt256 = c.plain()[steps.length - 1];
                ann = c.ann();
            }
        }
        double[] direct = new double[2];
        for (int k = 0; k < 2; k++) {
            direct[k] = SpikingClassifier.direct(k == 0 ? 4 : 8, SpikingClassifier.SpikeLoss.MEAN_OUTPUT,
                    SpikingClassifier.Surrogate.FAST_SIGMOID, 1, 300, 1);
        }
        System.out.printf("SNN direct LIF surrogate training (1 hidden): T 4 %.3f  T 8 %.3f%n", direct[0], direct[1]);
        assertTrue(ann > 0.85, "ANN " + ann);
        assertTrue(plainAt256 > ann - 0.05, "plain conversion matches the ANN at T = 256");
        assertTrue(qcfsMinusPlainAt4 > 0, "QCFS beats plain conversion at T = 4 in the deeper net");
    }
}
