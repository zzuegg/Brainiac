package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems;

import java.util.List;

import org.junit.jupiter.api.Test;

class KoopmanForecastTest {

    @Test
    void koopmanMethodsForecastVanDerPolAndLorenz() {
        DynamicsLearning.ForecastRow linear = DynamicsLearning.dmdAdvectionDiffusion(64, 60, 100);
        StringBuilder line = new StringBuilder(String.format(
                "8.3.10 forecast nMSE / valid steps: advection-diffusion DMD %.2e/%d;", linear.forecastError(),
                linear.validSteps()));
        for (int s = 0; s < 2; s++) {
            MechanicalSystems.Ode ode = s == 0 ? MechanicalSystems.vanDerPol(1) : MechanicalSystems.lorenz63(10, 28,
                    8.0 / 3);
            double[] x0 = s == 0 ? new double[] {2, 0} : new double[] {1, 1, 20};
            List<DynamicsLearning.ForecastRow> rows = DynamicsLearning.koopman(ode, x0, s == 0 ? 0.1 : 0.02, 600,
                    200, 300, 1);
            line.append(" [").append(ode.name()).append(']');
            rows.forEach(r -> line.append(String.format(" %s %.3g/%d;", r.model(), r.forecastError(),
                    r.validSteps())));
        }
        System.out.println(line);
        assertTrue(linear.forecastError() < 1e-3, line.toString());
    }
}
