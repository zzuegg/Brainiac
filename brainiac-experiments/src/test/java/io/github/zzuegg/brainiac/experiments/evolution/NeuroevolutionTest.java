package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experiments.evolution.BlackBoxOptimizers.Control;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuroevolutionTest {

    private static String format(Map<String, Double> m) {
        StringBuilder sb = new StringBuilder();
        m.forEach((k, v) -> sb.append(String.format("%s %.1f, ", k, v)));
        return sb.toString();
    }

    @Test
    void populationOptimizersEvolvePolicies() {
        Map<String, Double> cart = BlackBoxOptimizers.neuroevolution(Control.CART_POLE, 300, 2, 3);
        Map<String, Double> pend = BlackBoxOptimizers.neuroevolution(Control.PENDULUM, 800, 2, 3);
        System.out.printf("held-out return: cart-pole linear policy, 300 evaluations: %s| pendulum 3-8-1 MLP, 800 "
                + "evaluations: %s%n", format(cart), format(pend));
        assertTrue(cart.get("DE") > 300 && cart.get("PSO") > 300, "DE and PSO balance the pole");
        assertTrue(pend.get("PSO") > pend.get("random"), "PSO beats random search on the pendulum");
    }
}
