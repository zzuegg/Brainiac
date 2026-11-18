package io.github.zzuegg.brainiac.experiments.world;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class TokenWorldModelTest {

    @Test
    void tokenWorldModelAgainstMlpBaseline() {
        List<WorldModelShootout.TokenRow> rows = WorldModelShootout.tokenWorldModel(10, 30, 30, 1, 1);
        StringBuilder line = new StringBuilder("9.5.5 KeyDoor world models, token/cell acc, reward err, return:");
        for (WorldModelShootout.TokenRow r : rows) {
            line.append(String.format(" %s %.2f/%.2f %.3f %.2f;", r.model(), r.tokenAccuracy(), r.cellAccuracy(),
                    r.rewardError(), r.meanReturn()));
            assertTrue(Double.isFinite(r.meanReturn()), line.toString());
        }
        System.out.println(line);
        assertTrue(Double.isFinite(rows.get(0).tokenAccuracy()) && Double.isFinite(rows.get(0).rewardError()),
                line.toString());
    }
}
