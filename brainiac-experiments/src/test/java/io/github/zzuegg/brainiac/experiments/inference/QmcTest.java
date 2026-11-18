package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class QmcTest {

    @Test
    void randomisedQuasiMonteCarlo() {
        Map<String, double[]> q = Integration.qmc(new int[] {2, 8}, 11, 8, 1);
        System.out.printf("RQMC (13.5.10) {RMSE slope, rel RMSE @2048}: %s%n", NeuralLikelihoodTest.line(q));
        Map<String, double[]> v = Integration.variational(16, 200, 300, 2);
        System.out.printf("RQMC VI banana (13.5.10) {grad var, ELBO, ELBO@50 steps, last-20 ELBO}: %s%n",
                NeuralLikelihoodTest.line(v));
        assertTrue(q.get("gaussian d=2 SOBOL_OWEN")[0] < -0.8, "scrambled Sobol converges near n^-1 when smooth");
        assertTrue(q.get("gaussian d=8 SOBOL_OWEN")[1] < q.get("gaussian d=8 MC")[1], "RQMC beats MC");
        assertTrue(v.get("RQMC (Owen Sobol)")[0] < v.get("MC")[0], "RQMC lowers the gradient variance");
    }
}
