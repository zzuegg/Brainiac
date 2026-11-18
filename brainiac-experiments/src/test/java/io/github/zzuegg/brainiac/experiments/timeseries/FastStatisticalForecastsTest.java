package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class FastStatisticalForecastsTest {

    @Test
    void gridFitIsFastAndAccurate() {
        List<FastStatisticalForecasts.ForecastRow> rows = FastStatisticalForecasts.run(200, 96, 12, 5, 1);
        for (FastStatisticalForecasts.ForecastRow r : rows) {
            System.out.printf("12.2.9 %-40s %3d series: MASE %.3f, 80%% coverage %.3f, 95%% coverage %.3f,"
                    + " %8.2f ms/series%n", r.method(), r.series(), r.mase(), r.coverage80(), r.coverage95(),
                    r.millisPerSeries());
        }
        FastStatisticalForecasts.ForecastRow lbfgs = rows.get(0);
        FastStatisticalForecasts.ForecastRow grid = rows.get(1);
        assertTrue(grid.millisPerSeries() * 5 < lbfgs.millisPerSeries(), "grid fit is much faster");
        assertTrue(grid.mase() < lbfgs.mase() * 1.1, "grid fit is about as accurate");
        double naive = rows.stream().filter(r -> r.method().startsWith("seasonal naive")).findFirst().orElseThrow()
                .mase();
        double auto = rows.stream().filter(r -> r.method().startsWith("AutoETS")).findFirst().orElseThrow().mase();
        assertTrue(auto < naive, "AutoETS beats seasonal naive");
    }
}
