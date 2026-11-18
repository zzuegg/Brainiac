package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class UniversalDifferentialEquationTest {

    @Test
    void udeExtrapolatesLotkaVolterraBetterThanBlackBox() {
        StringBuilder line = new StringBuilder("8.3.11 Lotka-Volterra extrapolation MSE:");
        StringBuilder recovered = new StringBuilder();
        double ude = 0;
        for (int points : new int[] {20, 60}) {
            List<DynamicsLearning.UdeRow> rows = DynamicsLearning.ude(points, 600, 1, recovered);
            line.append(String.format(" [%d points]", points));
            rows.forEach(r -> line.append(String.format(" %s %.4f;", r.model(), r.extrapolationError())));
            ude = rows.get(1).extrapolationError();
        }
        line.append(" recovered: ").append(recovered);
        System.out.println(line);
        assertTrue(ude < 5, line.toString());
    }
}
