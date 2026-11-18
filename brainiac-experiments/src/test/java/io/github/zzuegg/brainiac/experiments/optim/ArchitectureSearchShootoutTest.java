package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.optim.ArchitectureSearchShootout.Method;
import io.github.zzuegg.brainiac.experiments.optim.ArchitectureSearchShootout.Outcome;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ArchitectureSearchShootoutTest {

    @Test
    void zeroCostProxiesRankArchitecturesAtInitialization() {
        double[] rho = ArchitectureSearchShootout.proxyCorrelations(24, 1);
        System.out.printf("Spearman with trained spiral accuracy (24 archs): grad-norm %.2f, SNIP %.2f, synflow %.2f, "
                + "NASWOT %.2f%n", rho[0], rho[1], rho[2], rho[3]);
        double best = Math.max(Math.max(rho[0], rho[1]), Math.max(rho[2], rho[3]));
        assertTrue(best > 0.3, "some proxy ranks architectures: " + java.util.Arrays.toString(rho));
    }

    @Test
    void enasAndZeroCostSearchCompareWithDartsAndRandomSearch() {
        Map<Method, Outcome> r = new EnumMap<>(Method.class);
        for (Method m : Method.values()) {
            r.put(m, ArchitectureSearchShootout.search(m, 1));
        }
        System.out.printf("spiral NAS test acc / search cost: ENAS %.3f/%d steps %s, NASWOT %.3f/%d evals %s | DARTS "
                + "%.3f/%d steps %s, random %.3f/%d steps %s%n", r.get(Method.ENAS).testAccuracy(),
                r.get(Method.ENAS).searchSteps(), r.get(Method.ENAS).architecture(),
                r.get(Method.NASWOT).testAccuracy(), r.get(Method.NASWOT).searchSteps(),
                r.get(Method.NASWOT).architecture(), r.get(Method.DARTS).testAccuracy(),
                r.get(Method.DARTS).searchSteps(), r.get(Method.DARTS).architecture(),
                r.get(Method.RANDOM).testAccuracy(), r.get(Method.RANDOM).searchSteps(),
                r.get(Method.RANDOM).architecture());
        assertTrue(r.get(Method.ENAS).testAccuracy() > 0.5, "ENAS finds a working cell");
        assertTrue(r.get(Method.NASWOT).testAccuracy() > 0.4, "NASWOT finds a working cell");
    }
}
