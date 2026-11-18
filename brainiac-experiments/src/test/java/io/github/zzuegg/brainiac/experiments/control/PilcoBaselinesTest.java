package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experiments.control.ClassicalControl.ModelFree;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class PilcoBaselinesTest {

    @Test
    void modelFreeBaselinesOnThePilcoTask() {
        ModelFree same = ClassicalControl.modelFree(540, 2);
        ModelFree more = ClassicalControl.modelFree(2700, 1);
        System.out.printf("cart-pole balance, test cost: at PILCO's 540 real steps SAC %.1f, PPO %.1f; at 2700 steps "
                + "SAC %.1f, PPO %.1f (zero force about 112)%n", same.sac(), same.ppo(), more.sac(), more.ppo());
        assertTrue(Double.isFinite(same.sac()) && Double.isFinite(more.ppo()), "the baselines run");
    }
}
