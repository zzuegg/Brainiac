package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class LagExtrapolationTest {

    @Test
    void stickBreakingLearnsExactLagsWithoutPositions() {
        List<LengthGeneralisation.Row> rows = LengthGeneralisation.lag(List.of("softmax RoPE", "softmax NoPE",
                "stick-breaking"), 200, 1);
        System.out.print(LengthGeneralisation.table("lag @ length", rows));
        double[] rope = rows.get(0).accuracy();
        double[] nope = rows.get(1).accuracy();
        double[] sb = rows.get(2).accuracy();
        System.out.printf("lag-2 accuracy at 4x training length: stick-breaking %.3f vs RoPE %.3f vs NoPE %.3f%n",
                sb[2], rope[2], nope[2]);
        assertTrue(sb[0] > nope[0] + 0.2, "stick-breaking has order without positions");
        assertTrue(sb[2] > nope[2] + 0.2);
        assertTrue(sb[2] > sb[0] - 0.1, "stick-breaking extrapolates");
    }
}
