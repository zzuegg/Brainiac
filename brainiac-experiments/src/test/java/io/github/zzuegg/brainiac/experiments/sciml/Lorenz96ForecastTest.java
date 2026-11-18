package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class Lorenz96ForecastTest {

    @Test
    void adaptiveNvarForecastsLorenz96() {
        List<DynamicsLearning.Lorenz96Row> rows = new ArrayList<>(DynamicsLearning.lorenz96Forecast(new int[] {8},
                new double[] {0, 0.05}, 4_000, 400, 0.01, 1_000, 1));
        rows.addAll(DynamicsLearning.lorenz96Forecast(new int[] {20}, new double[] {0.05}, 4_000, 400, 0.01, 1_000, 1));
        for (DynamicsLearning.Lorenz96Row r : rows) {
            System.out.printf("12.2.10 n=%-2d noise %.2f %-40s NMSE %9.3g, valid %.2f MTU, %6d params, %.2f s%n",
                    r.sites(), r.noise(), r.model(), r.forecastError(), r.validTime(), r.parameters(), r.seconds());
        }
        for (DynamicsLearning.Lorenz96Row r : rows) {
            if (r.sites() == 8 && r.noise() == 0 && (r.model().startsWith("adaptive") || r.model().startsWith("NG"))) {
                assertTrue(r.validTime() > 0.2, r.toString());
            }
        }
    }
}
