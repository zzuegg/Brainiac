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
}
