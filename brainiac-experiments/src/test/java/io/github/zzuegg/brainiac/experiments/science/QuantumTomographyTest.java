package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class QuantumTomographyTest {

    @Test
    void mleIsPhysicalAndShadowsBeatDirectEstimation() {
        Map<String, double[]> r = QuantumStates.tomography(3, new int[] {100, 1000, 10000}, 6, 3);
        System.out.printf("13.3.5 tomography (3-qubit TFIM state; shots {1e2, 1e3, 1e4}): %s%n", Lines.of(r));
        double[] mleNeg = r.get("min eigenvalue MLE");
        assertTrue(mleNeg[0] > -1e-9, Lines.of(r));
        assertTrue(r.get("infidelity MLE (R rho R) by shots")[2] < 0.05, Lines.of(r));
        double[] shadow = r.get("max error over 153 local Paulis, shots {1e3, 3e3, 1e4}: shadows");
        double[] direct = r.get("direct per-observable");
        assertTrue(shadow[2] < direct[2], Lines.of(r));
    }
}
