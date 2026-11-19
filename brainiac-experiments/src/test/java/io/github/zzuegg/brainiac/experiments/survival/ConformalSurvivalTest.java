package io.github.zzuegg.brainiac.experiments.survival;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ConformalSurvivalTest {

    @Test
    void conformalLowerBoundsCover() {
        for (boolean nonlinear : new boolean[] {false, true}) {
            Map<String, double[]> r = ConformalSurvival.conformal(1500, nonlinear, 0.4, false, 0.1, nonlinear ? 2 : 1);
            String setting = nonlinear ? "misspecified" : "well specified";
            r.forEach((k, v) -> System.out.printf("conformal LPB (%s Cox) %s: coverage %.3f, E[L]/E[T] %.3f%n",
                    setting, k, v[0], v[1]));
            assertTrue(r.get("TYPE_I_LPB")[0] > 0.85, "type-I LPB covers");
            assertTrue(r.get("NAIVE")[0] > 0.85, "naive LPB covers");
        }
        Map<String, double[]> f = ConformalSurvival.conformal(1500, true, 0.4, true, 0.1, 2);
        f.forEach((k, v) -> System.out.printf("conformal LPB (misspecified, survival forest) %s: coverage %.3f, "
                + "E[L]/E[T] %.3f%n", k, v[0], v[1]));
        assertTrue(f.get("TYPE_I_LPB")[0] > 0.85, "type-I LPB covers around a forest");
    }

    @Test
    void cureModel() {
        Map<String, Double> r = ConformalSurvival.cure(600, 3);
        r.forEach((k, v) -> System.out.printf("cure %s: %.3f%n", k, v));
        assertTrue(Math.abs(r.get("cure model (Weibull) cured fraction") - r.get("true cured fraction")) < 0.08,
                "cure fraction recovered");
    }

    @Test
    void competingRisks() {
        Map<String, Double> r = ConformalSurvival.competing(500, 4);
        r.forEach((k, v) -> System.out.printf("competing risks %s: %.3f%n", k, v));
        assertTrue(r.get("Aalen-Johansen CIF error") < r.get("1 - KM CIF error"), "AJ beats 1 - KM");
        assertTrue(r.get("Fine-Gray coefficient error") < r.get("cause-specific Cox coefficient error"),
                "Fine-Gray recovers subdistribution coefficients");
    }
}
