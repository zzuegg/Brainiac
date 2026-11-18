package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MessagePassingRolloutTest {

    @Test
    void pushforwardKeepsBurgersRolloutsOnTrack() {
        List<OperatorLearning.RolloutRow> rows = OperatorLearning.rollouts(24, 6, 40, 240, 1);
        StringBuilder line = new StringBuilder("8.3.6 Burgers rollouts, MSE / survival steps (train ms):");
        rows.forEach(r -> line.append(String.format(" %s %.4f/%.1f (%.0f);", r.model(), r.mse(), r.survival(),
                r.millis())));
        System.out.println(line);
        assertTrue(rows.get(0).survival() > 5, line.toString());
    }
}
