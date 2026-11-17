package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.train.BatchSizeSchedule;
import io.github.zzuegg.brainiac.experiments.optim.TrainingDynamicsExperiment.AveragingCurves;
import io.github.zzuegg.brainiac.experiments.optim.TrainingDynamicsExperiment.BudgetRun;
import io.github.zzuegg.brainiac.experiments.optim.TrainingDynamicsExperiment.Precision;
import io.github.zzuegg.brainiac.experiments.optim.TrainingDynamicsExperiment.SpikeRun;

import org.junit.jupiter.api.Test;

class TrainingDynamicsExperimentTest {

    @Test
    void spikeGuardSkipsOutlierBatches() {
        double clipped = 0;
        double guarded = 0;
        int skipped = 0;
        int outliers = 0;
        for (long seed = 1; seed <= 2; seed++) {
            SpikeRun c = TrainingDynamicsExperiment.spikes(false, 600, 10, seed);
            SpikeRun g = TrainingDynamicsExperiment.spikes(true, 600, 10, seed);
            clipped += c.finalMse() / 2;
            guarded += g.finalMse() / 2;
            skipped += g.skipped();
            outliers += g.outliers();
        }
        double clean = TrainingDynamicsExperiment.spikes(true, 600, 0, 1).finalMse();
        System.out.printf("SpikeGuard: MSE %.4f vs AdamW+clipping %.4f with outliers (clean run %.4f), "
                + "skipped %d of %d outlier batches%n", guarded, clipped, clean, skipped, outliers);
        assertEquals(outliers, skipped, "every outlier batch is caught, and nothing else");
        assertTrue(guarded < clipped, "skipping beats clipping the outliers into the moments");
    }

    @Test
    void lawaLeadsTheRawWeightsAndSwa() {
        AveragingCurves c = TrainingDynamicsExperiment.averaging(400, 50, 1);
        StringBuilder line = new StringBuilder();
        int lawaWins = 0;
        for (int i = 0; i < c.steps().length; i++) {
            line.append(String.format(" %d: %.3f/%.3f/%.3f/%.3f", c.steps()[i], c.lawa()[i], c.raw()[i], c.ema()[i],
                    c.swa()[i]));
            lawaWins += i > 0 && c.lawa()[i] < c.raw()[i] ? 1 : 0;
        }
        System.out.println("LAWA/raw/EMA/SWA held-out loss at step" + line);
        assertTrue(lawaWins == c.steps().length - 1, "LAWA beats the raw weights at every checkpoint after the first");
        int last = c.steps().length - 1;
        assertTrue(c.lawa()[last] < c.swa()[last], "and the all-history SWA average");
    }

    @Test
    void growingTheBatchReplacesLearningRateDecay() {
        long budget = 12800;
        BudgetRun cosine = TrainingDynamicsExperiment.batchRamp(BatchSizeSchedule.constant(32), true, 2e-2f, budget, 1);
        BudgetRun constant = TrainingDynamicsExperiment.batchRamp(BatchSizeSchedule.constant(32), false, 2e-2f, budget,
                1);
        BudgetRun ramp = TrainingDynamicsExperiment.batchRamp(BatchSizeSchedule.steps(32, 2, budget, 0.4, 0.7, 0.9),
                false, 2e-2f, budget, 1);
        System.out.printf("Batch ramp (12800 examples): loss %.3f in %d steps vs fixed batch + cosine %.3f in %d, "
                + "fixed batch + constant lr %.3f%n", ramp.finalLoss(), ramp.steps(), cosine.finalLoss(),
                cosine.steps(), constant.finalLoss());
        assertTrue(ramp.finalLoss() < constant.finalLoss(), "growing the batch cuts the noise like a decay");
        assertTrue(ramp.steps() < 0.7 * cosine.steps() && ramp.finalLoss() < 1.1f * cosine.finalLoss(),
                "with far fewer steps for nearly the same loss");
    }

    @Test
    void stochasticRoundingKeepsBf16WeightsLearning() {
        float fp32 = TrainingDynamicsExperiment.bf16(Precision.FLOAT32, 1e-3f, 1500, 1);
        float nearest = TrainingDynamicsExperiment.bf16(Precision.BF16_NEAREST, 1e-3f, 1500, 1);
        float stochastic = TrainingDynamicsExperiment.bf16(Precision.BF16_STOCHASTIC, 1e-3f, 1500, 1);
        System.out.printf("bf16 weights: stochastic rounding MSE %.3f vs round-to-nearest %.3f, fp32 %.3f%n", stochastic,
                nearest, fp32);
        assertTrue(stochastic < 0.2f * nearest, "round-to-nearest stalls, stochastic rounding does not");
        assertTrue(stochastic < 1.5f * fp32, "and gets close to float32");
    }
}
