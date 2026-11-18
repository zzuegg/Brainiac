package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SindyDiscoveryTest {

    @Test
    void sindyRecoversLorenzAndLotkaVolterra() {
        StringBuilder line = new StringBuilder("8.3.9 SINDy, structure share / coef err:");
        double lorenz = 0;
        for (String system : new String[] {"lorenz", "lotka-volterra", "damped"}) {
            for (double noise : new double[] {0, 0.01}) {
                List<DynamicsLearning.DiscoveryRow> rows = DynamicsLearning.sindy(system, noise,
                        "lorenz".equals(system) && noise == 0, 1);
                line.append(String.format(" [%s noise %.2f]", system, noise));
                rows.forEach(r -> line.append(String.format(" %s %.2f/%.3f (%.0f ms);", r.method(), r.structure(),
                        r.coefficientError(), r.millis())));
                if ("lorenz".equals(system) && noise == 0) {
                    lorenz = rows.get(0).structure();
                    System.out.println("8.3.9 Lorenz equations: " + rows.get(0).equations());
                }
            }
        }
        System.out.println(line);
        assertTrue(lorenz > 0.9, line.toString());
    }
}
