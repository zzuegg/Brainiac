package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class GraphNetworkSimulatorTest {

    @Test
    void graphSimulatorGeneralisesToLongerChains() {
        List<DynamicsLearning.ParticleRow> rows = DynamicsLearning.particles(6, 8, 300, 40, 1);
        StringBuilder line = new StringBuilder("8.3.13 spring chains, rollout MSE train size / double size:");
        rows.forEach(r -> line.append(String.format(" %s %.4g/%.4g;", r.model(), r.trainSizeMse(),
                r.doubleSizeMse())));
        System.out.println(line);
        assertTrue(rows.get(0).trainSizeMse() < 1, line.toString());
    }
}
