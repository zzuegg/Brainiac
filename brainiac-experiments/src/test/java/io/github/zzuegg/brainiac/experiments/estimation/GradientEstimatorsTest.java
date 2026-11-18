package io.github.zzuegg.brainiac.experiments.estimation;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.estimators.DiscreteGradients.Estimator;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class GradientEstimatorsTest {

    @Test
    void unbiasedEstimatorsKeepTheSignWhereStraightThroughFlipsIt() {
        Map<String, GradientEstimators.ToyRow> r = GradientEstimators.toy(-1, 12, 3000, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s logVar %.2f bias %.2f sign %.2f; ", k, v.logVariance(),
                v.relativeBias(), v.signCorrect())));
        System.out.printf("toy E[(b-0.499)^2] at 12 evals: %s%n", sb);
        assertTrue(r.get("STRAIGHT_THROUGH").signCorrect() == 0, sb.toString());
        for (String unbiased : new String[] {"REINFORCE", "RLOO", "ARM", "DISARM", "RODEO"}) {
            assertTrue(r.get(unbiased).signCorrect() == 1, sb.toString());
        }
        Map<String, GradientEstimators.ToyRow> q = GradientEstimators.interacting(12, 3000, 2);
        StringBuilder sq = new StringBuilder();
        q.forEach((k, v) -> sq.append(String.format("%s logVar %.2f bias %.2f; ", k, v.logVariance(),
                v.relativeBias())));
        System.out.printf("interacting toy at 12 evals: %s%n", sq);
        assertTrue(q.get("DISARM").logVariance() < q.get("ARM").logVariance(), sq.toString());
        assertTrue(q.get("RODEO").logVariance() < q.get("RLOO").logVariance() - 0.2, sq.toString());
        assertTrue(q.get("STRAIGHT_THROUGH").relativeBias() > 0.1, sq.toString());
    }

    @Test
    void vaeElbos() {
        Map<String, Double> b = GradientEstimators.binaryVae(List.of(Estimator.values()), 16, 250, 4, 2);
        Map<String, Double> c = GradientEstimators.categoricalVae(List.of(Estimator.REINFORCE, Estimator.RLOO,
                Estimator.STRAIGHT_THROUGH, Estimator.GUMBEL_ST), 4, 250, 4, 3);
        StringBuilder sb = new StringBuilder();
        b.forEach((k, v) -> sb.append(String.format("%s %.2f; ", k, v)));
        sb.append("| categorical: ");
        c.forEach((k, v) -> sb.append(String.format("%s %.2f; ", k, v)));
        System.out.printf("VAE train ELBO (nats): binary: %s%n", sb);
        assertTrue(b.get("RLOO") > b.get("REINFORCE"), sb.toString());
        assertTrue(b.get("DISARM") > b.get("REINFORCE"), sb.toString());
        assertTrue(c.get("RLOO") > c.get("REINFORCE") - 1, sb.toString());
    }
}
