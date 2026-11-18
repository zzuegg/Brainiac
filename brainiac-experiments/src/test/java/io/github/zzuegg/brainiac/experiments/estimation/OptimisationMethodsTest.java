package io.github.zzuegg.brainiac.experiments.estimation;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class OptimisationMethodsTest {

    private static String line(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append(' ');
            for (double x : v) {
                sb.append(String.format("%.3g ", x));
            }
            sb.append("; ");
        });
        return sb.toString();
    }

    private static String flat(Map<String, Double> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.3f; ", k, v)));
        return sb.toString();
    }

    @Test
    void hypergradientsTunePerFeatureDecayAndDataWeights() {
        Map<String, Double> wd = OptimisationMethods.weightDecay(25, 1);
        System.out.printf("per-feature decay, test MSE: %s%n", flat(wd));
        assertTrue(wd.get("IFT Neumann (10 terms)") < wd.get("grid search (global)"), flat(wd));
        assertTrue(wd.get("IFT CG (10 iterations)") < wd.get("grid search (global)"), flat(wd));
        Map<String, Double> rw = OptimisationMethods.dataReweighting(20, 2);
        System.out.printf("30%% flipped digits: %s%n", flat(rw));
        assertTrue(rw.get("AUROC / IFT data weights") > 0.6, flat(rw));
    }

    @Test
    void varianceReductionConvergesLinearlyOnConvexFiniteSums() {
        Map<String, double[]> r = OptimisationMethods.finiteSums(15, 3);
        System.out.printf("finite sums, f - f* and grads/n: %s%n", line(r));
        for (String p : new String[] {"least squares k=1e2", "logistic k=1e2"}) {
            assertTrue(r.get(p + " / SVRG")[0] < r.get(p + " / SGD")[0] / 10, line(r));
            assertTrue(r.get(p + " / SAGA")[0] < r.get(p + " / SGD")[0] / 10, line(r));
        }
    }

    @Test
    void frankWolfeVariantsAndProximalSolvers() {
        Map<String, double[]> r = OptimisationMethods.constrained(300, 4);
        System.out.printf("constrained, f - f* / support-or-rank / ms per it / test RMSE: %s%n", line(r));
        assertTrue(r.get("simplex LS / PAIRWISE_FW")[0] < r.get("simplex LS / FRANK_WOLFE")[0], line(r));
        assertTrue(r.get("simplex LS / AWAY_STEP_FW")[0] < r.get("simplex LS / FRANK_WOLFE")[0], line(r));
        assertTrue(r.get("trace-norm completion / FRANK_WOLFE")[2]
                < r.get("trace-norm completion / PROJECTED_GRADIENT")[2], line(r));
    }

    @Test
    void learnedOptimisersAgainstAdam() {
        Map<String, Double> r = OptimisationMethods.learnedOptimizers(30, 200, 5);
        System.out.printf("learned optimisers, final loss / tuned Adam: %s%n", flat(r));
        assertTrue(r.get("spiral MLP h8 (in-distribution) / Celo-lite") < 3, flat(r));
    }
}
