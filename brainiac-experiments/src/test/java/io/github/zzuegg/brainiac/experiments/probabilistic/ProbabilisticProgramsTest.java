package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class ProbabilisticProgramsTest {

    static String line(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k);
            for (double x : v) {
                sb.append(String.format(" %.3g", x));
            }
            sb.append("; ");
        });
        return sb.toString();
    }

    @Test
    void everyEngineRecoversTheLinearRegressionPosterior() {
        Map<String, double[]> r = ProbabilisticPrograms.linearRegression(1);
        System.out.printf("PPL linear regression {mean error, ESS/s}: %s%n", line(r));
        r.forEach((k, v) -> assertTrue(v[0] < 0.1, k + " " + v[0]));
    }

    @Test
    void discreteModelsMatchExactInference() {
        Map<String, double[]> cp = ProbabilisticPrograms.changePoint(2);
        System.out.printf("PPL change point {tau error, ESS/s}: %s%n", line(cp));
        Map<String, double[]> hmm = ProbabilisticPrograms.hmm(3);
        System.out.printf("PPL HMM {marginal error vs forward-backward, ESS/s}: %s%n", line(hmm));
        Map<String, double[]> sp = ProbabilisticPrograms.sprinkler(4);
        System.out.printf("PPL sprinkler {P(rain|wet) error vs VE, ESS/s}: %s%n", line(sp));
        assertTrue(cp.get("trace MH")[0] < 1.5, "change point MH");
        assertTrue(hmm.get("SMC")[0] < 0.05, "HMM SMC");
        assertTrue(hmm.get("SMC log Z error")[0] < 0.3, "HMM SMC evidence");
        sp.forEach((k, v) -> assertTrue(v[0] < (k.equals("trace MH") ? 0.05 : 0.03), "sprinkler " + k));
    }
}
