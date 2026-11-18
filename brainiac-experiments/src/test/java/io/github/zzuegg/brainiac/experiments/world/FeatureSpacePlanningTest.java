package io.github.zzuegg.brainiac.experiments.world;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class FeatureSpacePlanningTest {

    @Test
    void goalReachingInFrozenFeatureSpaces() {
        List<WorldModelShootout.ReachRow> rows = WorldModelShootout.goalReaching(400, 100, 150, 6, 1);
        StringBuilder line = new StringBuilder("9.5.7 goal reaching, success/final distance:");
        for (WorldModelShootout.ReachRow r : rows) {
            line.append(String.format(" %s %.2f/%.3f;", r.method(), r.success(), r.finalDistance()));
            assertTrue(Double.isFinite(r.finalDistance()), line.toString());
        }
        System.out.println(line);
    }
}
