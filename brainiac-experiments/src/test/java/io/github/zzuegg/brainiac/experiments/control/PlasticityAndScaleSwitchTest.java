package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PlasticityAndScaleSwitchTest {

    @Test
    void interventionsAfterTheGoalMoves() {
        Map<String, PlasticityAndScale.Outcome> r = PlasticityAndScale.taskSwitch(400, 1, 1);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s %.2f (dormant %.2f), ", k, v.ret(), v.dormant())));
        System.out.printf("W10 4.10 task switch, return on the moved goal after 400 + 400 steps: %s%n", b);
        for (PlasticityAndScale.Outcome o : r.values()) {
            assertTrue(Double.isFinite(o.ret()), o.toString());
        }
    }
}
