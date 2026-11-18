package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SirSurrogateTest {

    @Test
    void surrogateForecastsWithTheOde() {
        Map<String, double[]> r = Epidemics.surrogate(1000, new int[] {10, 13, 16, 19}, 1000, 3);
        System.out.printf("13.3.8 SIR surrogate rolling forecasts (SEIR truth, weekly): %s%n", Lines.of(r));
        double pinn = r.get("PINN surrogate {4-week MAE, mean MAE 1-4, cover50, cover90, s/window}")[1];
        double ode = r.get("exact ODE {4-week MAE, mean MAE 1-4, cover50, cover90, s/window}")[1];
        assertTrue(Double.isFinite(pinn) && Double.isFinite(ode), Lines.of(r));
        assertTrue(r.get("inside box s(u) MAE {PINN, MLP}")[0] < 0.1, Lines.of(r));
    }
}
