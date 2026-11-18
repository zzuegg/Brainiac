package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class CircuitShootoutTest {

    static String line(Map<String, double[]> r) {
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> {
            b.append(k).append(" {");
            for (int i = 0; i < v.length; i++) {
                b.append(String.format(i == 0 ? "%.3f" : ", %.3f", v[i]));
            }
            b.append("} | ");
        });
        return b.toString();
    }

    @Test
    void circuitsOnDigitsWithMissingInputs() {
        Map<String, double[]> r = CircuitShootout.digits(500, 200, 8, 3, 15, 1);
        System.out.printf("11.3.11 digits {test pixel LL, acc 50%% missing, acc complete}: %s%n", line(r));
        assertTrue(r.get("tensorised circuit (EM)")[0] > r.get("independent pixels")[0], line(r));
        assertTrue(r.get("tensorised circuit (EM)")[1] > r.get("mean imputation + logistic regression")[1] - 0.05,
                line(r));
        double[] m = CircuitShootout.marginals(40, 2);
        System.out.printf("11.3.11 marginals: max |circuit - brute force| %.2e, |circuit - truth| %.4f, "
                + "|mixture - truth| %.4f (log), ms/query circuit %.3f vs brute force %.3f%n", m[0], m[1], m[2], m[3],
                m[4]);
        assertTrue(m[0] < 1e-8, "marginals are exact");
    }

    @Test
    void squaredCircuitsAtMatchedSize() {
        Map<String, double[]> p = CircuitShootout.squared(true, 4, 200, 3);
        System.out.printf("11.3.12 planted cancellation {test LL, params}: %s%n", line(p));
        Map<String, double[]> d = CircuitShootout.squared(false, 4, 60, 4);
        System.out.printf("11.3.12 digit pixels {test LL, params}: %s%n", line(d));
        assertTrue(p.get("squared circuit (complex)")[0] > p.get("true distribution")[0] - 0.3, line(p));
        assertTrue(p.get("squared circuit (real)")[0] > p.get("true distribution")[0] - 0.5, line(p));
        assertTrue(Double.isFinite(d.get("Inception circuit (U=2, real)")[0]), line(d));
    }

    @Test
    void variationalMixturePrunesComponents() {
        Map<String, double[]> r = CircuitShootout.clustering(new int[] {3, 6, 10}, 400, 5);
        System.out.printf("11.3.13 clustering {ARI, |K err|, held-out LL, ms}: %s%n", line(r));
        assertTrue(r.get("VB-GMM (Dirichlet 0.01, K=16)")[0] > 0.85, line(r));
        assertTrue(r.get("VB-GMM (Dirichlet 0.01, K=16)")[3] < r.get("DP collapsed Gibbs")[3], line(r));
    }
}
