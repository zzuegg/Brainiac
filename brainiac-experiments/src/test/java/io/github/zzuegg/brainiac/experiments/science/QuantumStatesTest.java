package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class QuantumStatesTest {

    @Test
    void stochasticReconfigurationTrainsTheRbm() {
        Map<String, double[]> r = QuantumStates.nqs(8, 150, 1);
        System.out.printf("13.3.3 NQS N=8 (exact sums unless sampled): %s%n", Lines.of(r));
        double[] err = r.get("TFIM h=1 rel err {RBM SR, RBM Adam, RBM SR sampled, Jastrow SR, MLP SR}");
        assertTrue(err[0] < 1e-2, Lines.of(r));
        assertTrue(err[0] < r.get("baselines rel err {mean field, VQE HEA depth 2}")[0], Lines.of(r));
    }

    @Test
    void conditionalStateCoversTheCouplingRange() {
        Map<String, double[]> r = QuantumStates.foundation(8,
                new double[] {0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65, 1.85}, 60, 2);
        System.out.printf("13.3.4 foundation NQS TFIM N=8, h grid 0.25..1.85: %s%n", Lines.of(r));
        assertTrue(r.get("mean rel err {per-h, per-h quarter budget, conditional}")[2] < 0.05, Lines.of(r));
        double[] peak = r.get("chi_F peak h {model, exact}");
        assertTrue(Math.abs(peak[0] - peak[1]) < 0.5, Lines.of(r));
    }
}
