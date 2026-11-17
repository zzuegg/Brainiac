package io.github.zzuegg.brainiac.experiments.memory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class SleepForgettingTest {

    @Test
    void sleepingReducesForgetting() {
        SleepForgetting.Result awake = SleepForgetting.run(3, false, 1);
        SleepForgetting.Result asleep = SleepForgetting.run(3, true, 1);
        System.out.printf("sleep: forgetting awake %.3f, asleep %.3f; final accuracy awake %.3f, asleep %.3f%n",
                awake.forgetting(), asleep.forgetting(), awake.finalAccuracy(), asleep.finalAccuracy());
        assertTrue(awake.forgetting() > 0.4, "without sleep the old tasks are forgotten: " + awake.forgetting());
        assertTrue(asleep.forgetting() < 0.15, "with sleep: " + asleep.forgetting());
        assertTrue(asleep.finalAccuracy() > awake.finalAccuracy() + 0.2,
                asleep.finalAccuracy() + " vs " + awake.finalAccuracy());
    }
}
