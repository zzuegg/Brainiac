package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class FullBatchSecondOrderTest {

    private static String line(String title, Map<String, FullBatchSecondOrder.Row> r) {
        StringBuilder s = new StringBuilder(title);
        r.forEach((k, v) -> s.append(String.format(" %s %.2e/%d it/%d ms/target %d ms;", k, v.loss(), v.iterations(),
                v.millis(), v.millisToTarget())));
        System.out.println(s);
        return s.toString();
    }

    @Test
    void lbfgsAndHessianFreeFitANeuralFieldBetterThanFirstOrder() {
        Map<String, FullBatchSecondOrder.Row> r = FullBatchSecondOrder.neuralField(60, 3e-3, 1);
        String s = line("Neural field 20x20 (385 weights), MSE/iterations/ms/time-to-3e-3:", r);
        assertTrue(r.get("L-BFGS").loss() < r.get("Adam").loss(), s);
        assertTrue(r.get("Hessian-free").loss() < r.get("SGD momentum").loss(), s);
    }

    @Test
    void hessianFreeTrainsTheDeepAutoencoder() {
        Map<String, FullBatchSecondOrder.Row> r = FullBatchSecondOrder.autoencoder(25, 0.12, 2);
        String s = line("Digits autoencoder 64-32-8-32-64, MSE/iterations/ms/time-to-0.12:", r);
        assertTrue(r.get("Hessian-free").loss() < r.get("Adam").loss(), s);
        assertTrue(r.get("K-FAC").loss() < 0.2, s);
    }

    @Test
    void levenbergMarquardtBeatsAdamByOrdersOfMagnitude() {
        Map<String, FullBatchSecondOrder.Row> r = FullBatchSecondOrder.smallNetwork(40, 3);
        String s = line("Mackey-Glass 4-8-1 one-step, MSE/iterations/ms:", r);
        assertTrue(r.get("Levenberg-Marquardt").loss() * 10 < r.get("Adam").loss(), s);
    }
}
