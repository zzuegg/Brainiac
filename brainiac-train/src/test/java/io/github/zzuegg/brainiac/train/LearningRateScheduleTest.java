package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class LearningRateScheduleTest {

    @Test
    void stepHalvesEveryNEpochs() {
        LearningRateSchedule s = LearningRateSchedule.step(0.5f, 10);
        assertEquals(1f, s.rate(1, 1f), 1e-7f);
        assertEquals(1f, s.rate(10, 1f), 1e-7f);
        assertEquals(0.5f, s.rate(11, 1f), 1e-7f);
        assertEquals(0.25f, s.rate(21, 1f), 1e-7f);
    }

    @Test
    void exponentialDecaysEachEpoch() {
        LearningRateSchedule s = LearningRateSchedule.exponential(0.9f);
        assertEquals(0.1f, s.rate(1, 0.1f), 1e-7f);
        assertEquals(0.081f, s.rate(3, 0.1f), 1e-6f);
    }

    @Test
    void cosineGoesFromBaseToMin() {
        LearningRateSchedule s = LearningRateSchedule.cosine(11, 0.01f);
        assertEquals(1f, s.rate(1, 1f), 1e-6f);
        assertEquals(0.505f, s.rate(6, 1f), 1e-6f);
        assertEquals(0.01f, s.rate(11, 1f), 1e-6f);
        assertEquals(0.01f, s.rate(50, 1f), 1e-6f);
    }

    @Test
    void warmupRampsUpThenHandsOver() {
        LearningRateSchedule s = LearningRateSchedule.warmup(4, LearningRateSchedule.exponential(0.5f));
        assertEquals(0.25f, s.rate(1, 1f), 1e-6f);
        assertEquals(1f, s.rate(4, 1f), 1e-6f);
        assertEquals(LearningRateSchedule.exponential(0.5f).rate(6, 1f), s.rate(6, 1f), 1e-6f);
        assertThrows(IllegalArgumentException.class, () -> LearningRateSchedule.step(0.5f, 0));
    }

    @Test
    void shiftedWarmupStartsTheNextScheduleFresh() {
        LearningRateSchedule cosine = LearningRateSchedule.cosine(10, 0f);
        LearningRateSchedule s = LearningRateSchedule.warmup(3, cosine, true);
        assertEquals(1f / 3, s.rate(1, 1f), 1e-6f);
        assertEquals(cosine.rate(1, 1f), s.rate(4, 1f), 1e-6f);
        assertEquals(cosine.rate(10, 1f), s.rate(13, 1f), 1e-6f);
        assertEquals(LearningRateSchedule.warmup(3, cosine).rate(5, 1f),
                LearningRateSchedule.warmup(3, cosine, false).rate(5, 1f), 0f);
    }

    private static void follows(LearningRateSchedule s, double[] torch) {
        for (int i = 0; i < torch.length; i++) {
            assertEquals(torch[i], s.rate(i + 1, 1f), 1e-5, "step " + (i + 1));
        }
    }

    @Test
    void llmSchedulesFollowPyTorch() {
        // torch.optim.lr_scheduler.OneCycleLR(max_lr=1, total_steps=10)
        follows(LearningRateSchedule.oneCycle(10), new double[] {0.04, 0.52, 1.0, 0.950485, 0.811746, 0.611262,
            0.388742, 0.188258, 0.049519, 4e-06});
        // CosineAnnealingWarmRestarts(T_0=3, T_mult=2, eta_min=0.1)
        follows(LearningRateSchedule.cosineWithRestarts(3, 2, 0.1f), new double[] {1.0, 0.775, 0.325, 1.0, 0.939711,
            0.775, 0.55, 0.325, 0.160289, 1.0});
        follows(LearningRateSchedule.linear(5, 0), new double[] {1, 0.75, 0.5, 0.25, 0, 0});
        follows(LearningRateSchedule.warmupStableDecay(2, 2, 4, 0, false), new double[] {0.5, 1, 1, 1, 0.75, 0.5,
            0.25, 0, 0});
        assertEquals(1 - Math.sqrt(0.25), LearningRateSchedule.warmupStableDecay(0, 0, 4, 0, true).rate(1, 1f), 1e-6);
        follows(LearningRateSchedule.inverseSqrt(4), new double[] {0.25, 0.5, 0.75, 1, Math.sqrt(0.8)});
    }
}
