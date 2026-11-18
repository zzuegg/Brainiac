package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class NeuralSdeTest {

    @Test
    void neuralSdeMatchesStochasticMarginals() {
        StringBuilder line = new StringBuilder("8.3.12 NLL / W2 / switch rate:");
        double ou = 0;
        for (boolean well : new boolean[] {false, true}) {
            List<DynamicsLearning.SdeRow> rows = DynamicsLearning.stochastic(well, 1000, 200, 1);
            line.append(well ? " [double well]" : " [OU]");
            rows.forEach(r -> line.append(String.format(" %s %.3f/%.3f/%.3f;", r.model(), r.nll(), r.wasserstein(),
                    r.switchingRate())));
            if (!well) {
                ou = rows.get(1).wasserstein();
            }
        }
        double[] latent = DynamicsLearning.latentSde(16, 150, 1);
        line.append(String.format(" [latent SDE] W2 %.3f gen rate %.3f true rate %.3f loss %.3f", latent[0],
                latent[1], latent[2], latent[3]));
        System.out.println(line);
        assertTrue(ou < 0.5, line.toString());
    }
}
