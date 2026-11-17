package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ImplicitQuantileMazeTest {

    @Test
    void iqnSolvesTheMazeLikeTheOtherDistributionalDqns() {
        Map<String, Double> r = RiskSensitiveDqn.maze(2000, 2);
        System.out.printf("W6 3.5 maze greedy return: IQN %.2f vs C51 %.2f, QR-DQN %.2f%n", r.get("IQN"), r.get("C51"),
                r.get("QR-DQN"));
        assertTrue(r.get("IQN") > 0.5, r.toString());
    }
}
