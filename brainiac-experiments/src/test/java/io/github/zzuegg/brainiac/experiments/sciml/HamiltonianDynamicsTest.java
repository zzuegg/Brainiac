package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems;

import java.util.List;

import org.junit.jupiter.api.Test;

class HamiltonianDynamicsTest {

    @Test
    void hamiltonianNetworkConservesPendulumEnergy() {
        List<DynamicsLearning.EnergyRow> rows = DynamicsLearning.hamiltonian(MechanicalSystems.pendulum(1), 200, 0.05,
                400, 300, 1);
        StringBuilder line = new StringBuilder("8.3.8 pendulum, rollout MSE / energy drift:");
        rows.forEach(r -> line.append(String.format(" %s %.4f/%.3f;", r.model(), r.rolloutMse(), r.energyDrift())));
        System.out.println(line);
        assertTrue(rows.get(0).energyDrift() < 0.5, line.toString());
    }
}
