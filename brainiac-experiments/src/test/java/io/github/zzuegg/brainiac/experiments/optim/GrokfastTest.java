package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.train.Grokfast;
import io.github.zzuegg.brainiac.train.Optimizer;

import org.junit.jupiter.api.Test;

class GrokfastTest {

    @Test
    void grokfastAgainstAdamWWithWeightDecay() {
        GrokkingExperiment.Curve plain = GrokkingExperiment.run(23, 0.8, 128, 1000,
                Optimizer.adamW(1e-2f, 0.9f, 0.98f, 1e-8f, 1f), 50, 1);
        GrokkingExperiment.Curve fast = GrokkingExperiment.run(23, 0.8, 128, 1000,
                Grokfast.ema(0.98f, 5f).wrap(Optimizer.adamW(1e-2f, 0.9f, 0.98f, 1e-8f, 1f)), 50, 1);
        System.out.printf("Grokfast-EMA: test acc > 0.99 at step %d vs AdamW+wd %d (memorised at %d vs %d), "
                + "final test acc %.3f vs %.3f%n", fast.generalisedAt(0.99), plain.generalisedAt(0.99),
                fast.memorisedAt(), plain.memorisedAt(), fast.testAccuracy().getLast(), plain.testAccuracy().getLast());
        assertTrue(fast.memorisedAt() > 0 && fast.testAccuracy().getLast() > 0.9, "Grokfast memorises and generalises");
        assertTrue(plain.testAccuracy().getLast() > 0.9, "so does the baseline in this budget");
    }
}
