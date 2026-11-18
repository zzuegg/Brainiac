package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class QuantumAlgorithmsTest {

    @Test
    void hamiltonianVariationalAnsatzSolvesTheIsingChain() {
        Map<String, double[]> r = QuantumAlgorithms.vqe(6, new double[] {0.5, 1, 2}, 3, 1);
        System.out.printf("13.3.1 VQE TFIM N=6 h={0.5,1,2}, depth 3: %s%n", Lines.of(r));
        assertTrue(r.get("rel err HVA (L-BFGS)")[1] < 1e-3, Lines.of(r));
        assertTrue(r.get("rel err HVA (L-BFGS)")[1] < r.get("rel err mean field")[1], Lines.of(r));
        assertTrue(r.get("max |adjoint - parameter shift|")[0] < 1e-9, Lines.of(r));
    }

    @Test
    void qaoaRatioRisesWithDepth() {
        Map<String, double[]> r = QuantumAlgorithms.qaoa(10, 8, 4, 400, 2);
        System.out.printf("13.3.2 QAOA MaxCut 3-regular n=10, p=1..4: %s%n", Lines.of(r));
        double[] interp = r.get("INTERP ratio by p");
        assertTrue(interp[3] > interp[0], Lines.of(r));
        assertTrue(interp[3] > 0.8, Lines.of(r));
        assertTrue(r.get("classical ratio {SB, p-bit, greedy+LS, random}")[0] > 0.99, Lines.of(r));
    }
}
