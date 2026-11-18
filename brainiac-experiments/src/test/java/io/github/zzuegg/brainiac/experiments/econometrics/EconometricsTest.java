package io.github.zzuegg.brainiac.experiments.econometrics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.PanelStudies;

import org.junit.jupiter.api.Test;

import java.util.Map;

class EconometricsTest {

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
    void panelEstimatorsBeatTwfeUnderFactorConfounding() {
        Map<String, double[]> s = PanelShootout.factor(PanelShootout.Design.SINGLE_SHIFTED, 10, 4, 1);
        System.out.printf("11.3.6 single unit, shifted loadings {bias, RMSE, placebo coverage}: %s%n", line(s));
        Map<String, double[]> c = PanelShootout.factor(PanelShootout.Design.NO_CONVEX_FIT, 10, 4, 100);
        System.out.printf("11.3.6 no convex fit {bias, RMSE, placebo coverage}: %s%n", line(c));
        assertTrue(s.get("synthetic DiD")[1] < s.get("TWFE DiD")[1], line(s));
        assertTrue(s.get("TROP (LOO-tuned)")[1] < s.get("TWFE DiD")[1], line(s));
        assertTrue(c.get("synthetic DiD")[1] < c.get("synthetic control")[1], line(c));
    }

    @Test
    void staggeredAdoptionAvoidsForbiddenComparisons() {
        Map<String, double[]> g = PanelShootout.staggered(PanelStudies.Effects.GROWING, 0, 15, 40, 1);
        System.out.printf("11.3.7 growing effects {bias, RMSE, band coverage, profile RMSE, forbidden weight}: %s%n",
                line(g));
        Map<String, double[]> c = PanelShootout.staggered(PanelStudies.Effects.CONSTANT, 0, 10, 0, 50);
        System.out.printf("11.3.7 constant effects {bias, RMSE, band coverage, profile RMSE, forbidden weight}: %s%n",
                line(c));
        assertTrue(Math.abs(g.get("Callaway-Sant'Anna (unconditional)")[0])
                < Math.abs(g.get("TWFE DiD (Bacon: forbidden weight)")[0]), line(g));
        assertTrue(Math.abs(g.get("Sun-Abraham")[0]) < Math.abs(g.get("TWFE DiD (Bacon: forbidden weight)")[0]),
                line(g));
    }

    @Test
    void robustBiasCorrectionRestoresCoverage() {
        Map<String, double[]> s = DiscontinuityDesigns.sharp(500, 200, 1);
        System.out.printf("11.3.8 sharp RD {bias, RMSE, coverage}: %s%n", line(s));
        Map<String, double[]> f = DiscontinuityDesigns.fuzzy(2000, 60, 2);
        System.out.printf("11.3.8 fuzzy RD {bias, RMSE, coverage}: %s%n", line(f));
        double[] d = DiscontinuityDesigns.density(1000, 0.5, 100, 3);
        double[] h = DiscontinuityDesigns.bandwidths(500, 30, 1);
        System.out.printf("11.3.8 mean bandwidths: IK h %.3f, CCT h %.3f, CCT b %.3f%n", h[0], h[1], h[2]);
        System.out.printf("11.3.8 McCrary test: size %.3f, power at 50%% manipulation %.3f%n", d[0], d[1]);
        assertTrue(s.get("local linear, CCT h, robust bias-corrected CI")[2] > 0.85, line(s));
        assertTrue(s.get("local linear, CCT h, robust bias-corrected CI")[2]
                >= s.get("local linear, CCT h, conventional CI")[2] - 0.02, line(s));
        assertTrue(d[1] > d[0] + 0.3, "power " + d[1] + " size " + d[0]);
    }

    @Test
    void balancingWeightsAndSensitivity() {
        Map<String, double[]> w = ObservationalAdjustment.weighting(true, 1, 500, 30, 1);
        System.out.printf("11.3.9 Kang-Schafer misspecified {bias, RMSE, max SMD, eff. controls}: %s%n", line(w));
        Map<String, double[]> c = ObservationalAdjustment.weighting(false, 1, 500, 20, 2);
        System.out.printf("11.3.9 Kang-Schafer correct {bias, RMSE, max SMD, eff. controls}: %s%n", line(c));
        assertTrue(w.get("entropy balancing weights")[1] < w.get("ipw weights")[1], line(w));
        assertTrue(w.get("entropy balancing weights")[1] < w.get("naive difference")[1], line(w));
        StringBuilder b = new StringBuilder();
        double[] robust = ObservationalAdjustment.sensitivity(20, 0.3, 1000, 5, 10);
        double[] confounded = ObservationalAdjustment.sensitivity(0, 1, 1000, 5, 20);
        for (double[] row : new double[][] {robust, confounded}) {
            b.append(String.format("{OLS %.2f, CI excl. 0 %.2f, EB %.2f, E-value %.2f, RV %.3f, adj(k=1) %.2f, "
                    + "k* %.2f, Gamma* %.2f} ", row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]));
        }
        b.append("| null studies, k* by planted s:");
        for (double s : new double[] {0.25, 0.5, 1, 2}) {
            b.append(String.format(" s=%.2f->%.2f", s, ObservationalAdjustment.sensitivity(0, s, 1000, 3, 30)[6]));
        }
        System.out.printf("11.3.10 sensitivity: robust (effect 20, s 0.3) vs null (effect 0, s 1): %s%n", b);
        assertTrue(confounded[1] > 0.5, "the null study's CI alone does not flag it: " + b);
        assertTrue(robust[5] > 0 && confounded[5] <= 0, b.toString());
    }
}
