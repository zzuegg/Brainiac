package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.nn.Model;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertTrue;

class FractalDropPathTest {
    @Test
    void dropPathMakesEveryColumnAStandaloneClassifier() {
        Model with = FractalDropPath.model(3, 0.15, 0.5, 1);
        Model without = FractalDropPath.model(3, 0, 0, 1);
        FractalDropPath.train(with, 500, 2);
        FractalDropPath.train(without, 500, 2);
        double[] a = FractalDropPath.columnAccuracies(with, 400, 3);
        double[] b = FractalDropPath.columnAccuracies(without, 400, 3);
        System.out.println("drop-path " + Arrays.toString(a) + ", none " + Arrays.toString(b));
        assertTrue(a[3] > 0.9 && b[3] > 0.9, "joined blocks should both learn");
        double worstWith = Math.min(a[0], Math.min(a[1], a[2]));
        double worstWithout = Math.min(b[0], Math.min(b[1], b[2]));
        assertTrue(worstWith > 0.85, "every drop-path column alone " + worstWith);
        assertTrue(worstWithout < worstWith - 0.1, "without drop-path some column should fail alone");
    }
}
