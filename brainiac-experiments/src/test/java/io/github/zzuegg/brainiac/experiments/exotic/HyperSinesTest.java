package io.github.zzuegg.brainiac.experiments.exotic;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class HyperSinesTest {
    @Test
    void generatedNetworksFitUnseenTasks() {
        HyperSines h = HyperSines.of(5e-3f, 1);
        List<double[]> unseen = List.of(new double[] {1.37, 0.4}, new double[] {0.6, 2.9}, new double[] {1.9, 5.1});
        float before = h.error(unseen, 5);
        h.train(400);
        float after = h.error(unseen, 5);
        // the variance of A sin(x + phi) over these tasks is mean(A^2)/2 ~ 0.99
        System.out.printf("hypernetwork unseen-task MSE %.4f -> %.4f%n", before, after);
        assertTrue(after < 0.05f, "MSE " + after);
        assertTrue(after < 0.05f * before);
    }
}
