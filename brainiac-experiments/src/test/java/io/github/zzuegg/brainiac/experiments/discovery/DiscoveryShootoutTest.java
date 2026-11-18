package io.github.zzuegg.brainiac.experiments.discovery;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.DiscoveryBenchmarks;

import org.junit.jupiter.api.Test;

import java.util.Map;

class DiscoveryShootoutTest {

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
    void pcWithOrientationOnGaussianData() {
        Map<String, double[]> r = DiscoveryShootout.constraintBased(10, 1000, false, 2, 1);
        System.out.printf("11.3.1 PC, linear-Gaussian d=10 {SHD vs CPDAG, orient P, orient R, ms}: %s%n", line(r));
        assertTrue(r.get("PC + Meek")[0] < r.get("empty graph")[0], line(r));
        assertTrue(r.get("PC + Meek")[0] <= r.get("PC skeleton only")[0], line(r));
        Map<String, double[]> k = DiscoveryShootout.constraintBased(6, 250, true, 1, 2);
        System.out.printf("11.3.1 PC (kernel CI), nonlinear d=6: %s%n", line(k));
        assertTrue(k.get("PC + Meek")[0] < k.get("empty graph")[0], line(k));
    }

    @Test
    void fciWithHiddenConfounders() {
        Map<String, double[]> r = DiscoveryShootout.hidden(7, 2, 2000, 3, 3);
        System.out.printf("11.3.1 FCI, 2 hidden confounders {arrow P, arrow R, tail P, dist to oracle PAG}: %s%n",
                line(r));
        assertTrue(r.get("FCI")[0] >= r.get("PC (CPDAG read as marks)")[0] - 0.05, line(r));
        assertTrue(r.get("FCI")[3] <= r.get("PC (CPDAG read as marks)")[3], line(r));
    }

    @Test
    void lingamOnNonGaussianAndGaussianData() {
        Map<String, double[]> r = DiscoveryShootout.nonGaussian(6, 1000, DiscoveryBenchmarks.Noise.UNIFORM, true, 2, 4);
        System.out.printf("11.3.2/3 uniform noise, standardised {SHD vs DAG, ordered pairs, ms}: %s%n", line(r));
        assertTrue(r.get("DirectLiNGAM")[1] > 0.85, line(r));
        assertTrue(r.get("DirectLiNGAM")[0] <= r.get("NOTEARS")[0], line(r));
        Map<String, double[]> g = DiscoveryShootout.nonGaussian(6, 1000, DiscoveryBenchmarks.Noise.GAUSSIAN, false, 2,
                5);
        System.out.printf("11.3.2/3 Gaussian noise, raw equal variances: %s%n", line(g));
        assertTrue(g.get("TriOpt")[1] > 0.85, line(g));
    }

    @Test
    void scoreOrderingOnNonlinearModels() {
        Map<String, double[]> r = DiscoveryShootout.nonlinearOrdering(6, 300, 2, 6);
        System.out.printf("11.3.3 SCORE, nonlinear ANM d=6 {order divergence, SHD, ms}: %s%n", line(r));
        assertTrue(r.get("SCORE")[0] <= r.get("random order")[0], line(r));
    }

    @Test
    void bivariateCauseEffect() {
        for (DiscoveryBenchmarks.PairKind kind : DiscoveryBenchmarks.PairKind.values()) {
            Map<String, double[]> r = DiscoveryShootout.pairs(kind, 30, 200, 7);
            System.out.printf("11.3.4 pairs %s {accuracy, AUC}: %s%n", kind, line(r));
            if (kind == DiscoveryBenchmarks.PairKind.ADDITIVE) {
                assertTrue(r.get("ANM (kernel ridge + HSIC)")[0] > 0.65, line(r));
            }
            if (kind == DiscoveryBenchmarks.PairKind.DETERMINISTIC) {
                assertTrue(r.get("IGCI slope")[0] > 0.65, line(r));
            }
        }
    }
}
