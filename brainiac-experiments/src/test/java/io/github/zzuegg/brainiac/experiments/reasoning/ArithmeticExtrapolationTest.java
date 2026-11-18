package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticExtrapolation.Architecture;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticExtrapolation.Operation;

import org.junit.jupiter.api.Test;

class ArithmeticExtrapolationTest {

    private static double[][] all(Operation op) {
        StringBuilder line = new StringBuilder(op + " relative error, train range / [10, 1000]:");
        double[][] e = new double[Architecture.values().length][];
        for (Architecture a : Architecture.values()) {
            e[a.ordinal()] = ArithmeticExtrapolation.run(a, op, 1500, 1);
            line.append(String.format(" %s %.3g/%.3g;", a, e[a.ordinal()][0], e[a.ordinal()][1]));
        }
        System.out.println(line);
        return e;
    }

    @Test
    void nacExtrapolatesAddition() {
        double[][] e = all(Operation.ADD);
        double nac = e[Architecture.NAC.ordinal()][1];
        assertTrue(nac < 0.05, "the NAC learns addition and keeps it far outside the training range");
        assertTrue(nac < 2 * e[Architecture.NAC.ordinal()][0], "its relative error does not grow with the inputs");
    }

    @Test
    void naluExtrapolatesMultiplication() {
        double[][] e = all(Operation.MULTIPLY);
        double nalu = e[Architecture.NALU.ordinal()][1];
        assertTrue(nalu < e[Architecture.MLP_RELU.ordinal()][1] && nalu < e[Architecture.MLP_SILU.ordinal()][1],
                "the NALU extrapolates products better than MLPs");
    }
}
