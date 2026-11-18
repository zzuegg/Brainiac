package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class BayesianQuadratureTest {

    @Test
    void quadratureHerdingAndEvidence() {
        Map<String, double[]> q = Integration.quadrature(new int[] {1, 2, 4, 8}, 50, 5, 1);
        System.out.printf("BQ n=50 (13.5.11) {rel err, coverage or nodes}: %s%n", NeuralLikelihoodTest.line(q));
        Map<String, double[]> h = Integration.herding(2, new int[] {16, 32, 64, 128}, 2);
        System.out.printf("Herding MMD d=2 (13.5.11) {n=16, 32, 64, 128, slope}: %s%n", NeuralLikelihoodTest.line(h));
        Map<String, double[]> e = Integration.evidence(3, 100, 3);
        System.out.printf("Evidence d=3 (13.5.11) {|log Z err|, evaluations}: %s%n", NeuralLikelihoodTest.line(e));
        assertTrue(q.get("smooth d=2 BQ (Sobol points)")[0] < 0.1 * q.get("smooth d=2 MC")[0],
                "BQ beats MC by an order of magnitude on a smooth integrand");
        assertTrue(h.get("herding")[4] < h.get("i.i.d.")[4], "herding's MMD falls faster than i.i.d.");
    }
}
