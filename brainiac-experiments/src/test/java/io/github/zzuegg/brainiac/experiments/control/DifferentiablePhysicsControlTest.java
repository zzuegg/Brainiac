package io.github.zzuegg.brainiac.experiments.control;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DifferentiablePhysicsControlTest {

    @Test
    void gradientsThroughPhysicsBeatBlackBoxSearch() {
        DifferentiablePhysicsControl.Result r = DifferentiablePhysicsControl.run(2000, 1);
        System.out.printf("mass-spring reach, 2000 episodes each: final distance BPTT %.3f vs CEM %.3f vs PPO %.3f "
                + "(untrained %.3f)%n", r.bptt(), r.cem(), r.ppo(), r.untrained());
        assertTrue(r.bptt() < 0.5 * r.untrained(), "BPTT learns the task");
        assertTrue(r.bptt() < r.cem(), "BPTT beats CEM");
        assertTrue(r.bptt() < r.ppo(), "BPTT beats PPO");
    }
}
