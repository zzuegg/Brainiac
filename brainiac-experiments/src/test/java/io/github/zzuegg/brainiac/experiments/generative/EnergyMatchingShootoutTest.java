package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EnergyMatchingShootoutTest {

    @Test
    void oneEnergyGivesSamplesAndADensity() {
        Map<String, Double> m = EnergyMatchingShootout.run(r -> PointClouds.moons(128, 0.05, r), 600, 20, 1);
        System.out.printf("moons, 20 sampler steps: sliced W EnM %.3f; baselines flow matching %.3f, EqM %.3f | "
                + "NLL nats EnM (grid-normalised exp(-V/T)) %.3f; baseline flow matching (exact ODE) %.3f%n",
                m.get("EnM SW"), m.get("flow matching SW"), m.get("EqM SW"), m.get("EnM NLL"),
                m.get("flow matching NLL"));
        assertTrue(m.get("EnM SW") < 0.25, "energy descent lands on the moons");
        // the Boltzmann density is proper but, at this budget, behind flow matching's exact likelihood
        assertTrue(m.get("EnM NLL") < 2.6, "exp(-V/T) is a usable density");
    }
}
