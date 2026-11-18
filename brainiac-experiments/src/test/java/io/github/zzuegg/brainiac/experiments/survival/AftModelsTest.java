package io.github.zzuegg.brainiac.experiments.survival;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SurvivalData;
import io.github.zzuegg.brainiac.experimental.survival.Weibull;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.SplittableRandom;

class AftModelsTest {

    static String line(Map<String, double[]> r) {
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s {%.3f, %.3f, %.3f} | ", k, v[0], v[1], v[2])));
        return b.toString();
    }

    @Test
    void weibullMleRecoversParameters() {
        SplittableRandom r = new SplittableRandom(1);
        Weibull truth = new Weibull(0.2, 1.7);
        double[] t = new double[4000];
        int[] e = new int[t.length];
        for (int i = 0; i < t.length; i++) {
            double event = truth.quantile(r.nextDouble());
            double censor = 8 * r.nextDouble();
            t[i] = Math.min(event, censor);
            e[i] = event <= censor ? 1 : 0;
        }
        Weibull fit = Weibull.fit(t, e);
        assertEquals(truth.k(), fit.k(), 0.1);
        assertEquals(truth.lambda(), fit.lambda(), 0.03);
        assertEquals(truth.density(1.3), truth.hazard(1.3) * truth.survival(1.3), 1e-12);
    }

    @Test
    void aftModelsAgainstCoxAndDeepModels() {
        Map<String, double[]> p = SurvivalShootout.aft(false, SurvivalData.Risk.LINEAR, 0.4, 1000, 300, 0, 1);
        System.out.printf("11.3.17 Weibull PH, linear, 40%% censored {C, IBS, |log median err|}: %s%n", line(p));
        Map<String, double[]> c = SurvivalShootout.aft(true, null, 0.3, 1000, 300, 20, 2);
        System.out.printf("11.3.17 crossing hazards, 30%% censored {td-C, IBS, |log median err|}: %s%n", line(c));
        assertTrue(Math.abs(p.get("weibull AFT")[0] - p.get("Cox")[0]) < 0.02, line(p));
        assertTrue(p.get("weibull AFT")[2] < p.get("Cox")[2], line(p));
        assertTrue(c.get("Deep Survival Machines (4 Weibulls)")[0] > c.get("weibull AFT")[0], line(c));
    }
}
