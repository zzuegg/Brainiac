package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.data.ChaoticSeries;
import io.github.zzuegg.brainiac.experimental.nn.EchoStateReservoir;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ReservoirForecastTest {
    // Lorenz: largest Lyapunov exponent ~0.9, so one Lyapunov time is ~1.1 time units = 44 steps of 0.025
    private static final int LYAPUNOV_STEPS = 44;

    @Test
    void nextGenerationForecastsLorenzForSeveralLyapunovTimes() {
        double[][] s = ChaoticSeries.lorenz(1400, 0.025, 2000);
        int train = 600;
        ReservoirForecast.Forecast f = ReservoirForecast.nextGeneration(s, train, 800, 2, 1, 1e-2);
        int valid = ReservoirForecast.validSteps(f.predicted(), s, train, 0.3);
        System.out.printf("NG-RC train nrmse %.2e, valid %d steps%n", f.trainingError(), valid);
        assertTrue(f.trainingError() < 1e-3, "one-step error " + f.trainingError());
        assertTrue(valid > 3 * LYAPUNOV_STEPS, "valid steps " + valid);
    }

    @Test
    void echoStateNetworkForecastsLorenz() {
        double[][] s = ChaoticSeries.lorenz(2400, 0.025, 2000);
        int train = 2000;
        EchoStateReservoir r = EchoStateReservoir.of(3, 300, 0.9, 0.5, 0.6, 0.03, 7);
        ReservoirForecast.Forecast f = ReservoirForecast.echoState(s, train, 400, r, 100, 1e-6);
        int valid = ReservoirForecast.validSteps(f.predicted(), s, train, 0.3);
        System.out.printf("ESN train nrmse %.2e, valid %d steps%n", f.trainingError(), valid);
        assertTrue(f.trainingError() < 0.01, "one-step error " + f.trainingError());
        assertTrue(valid > 2 * LYAPUNOV_STEPS, "valid steps " + valid);
    }

    @Test
    void echoStateNetworkForecastsMackeyGlass() {
        double[] mg = ChaoticSeries.mackeyGlass(2400, 17, 1, 500);
        double[][] s = new double[mg.length][];
        for (int i = 0; i < mg.length; i++) {
            s[i] = new double[] {mg[i]};
        }
        int train = 2000;
        EchoStateReservoir r = EchoStateReservoir.of(1, 200, 0.95, 0.5, 0.5, 0.05, 3);
        ReservoirForecast.Forecast f = ReservoirForecast.echoState(s, train, 300, r, 100, 1e-7);
        int valid = ReservoirForecast.validSteps(f.predicted(), s, train, 0.2);
        System.out.printf("MG ESN train nrmse %.2e, valid %d steps%n", f.trainingError(), valid);
        assertTrue(valid > 150, "valid steps " + valid);
    }
}
