package io.github.zzuegg.brainiac.experimental.align;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/** Wave 9 WP2: closed-form checks of the alignment building blocks. */
class AlignBlocksTest {

    @Test
    void ensembleCombination() {
        float[] col = {1, 2, 3};
        assertEquals(2, RewardEnsemble.combine(col, RewardEnsemble.Mode.MEAN, 0), 1e-6);
        assertEquals(1, RewardEnsemble.combine(col, RewardEnsemble.Mode.WORST, 0), 1e-6);
        assertEquals(1, RewardEnsemble.combine(col, RewardEnsemble.Mode.UNCERTAINTY, 1), 1e-6);
        System.out.printf("align: ensemble mean/worst/uncertainty of {1,2,3} = 2/1/1%n");
    }

    @Test
    void bestOfNQuantiles() {
        double[] q = BestOfNDistiller.quantiles(new float[] {-10, 2, 10}, new float[] {1, 2, 3});
        assertTrue(q[0] > 0 && q[0] < q[1] && q[1] < q[2] && q[2] < 1);
        assertEquals(2.0 / 4, q[1], 1e-9);
    }

    @Test
    void historicalVarianceRanking() {
        double[][] h = {{0.5, 0.5, 0.5}, {0, 1, 0}, {0.2, 0.4, 0.6}};
        double[] v = SingleExampleSelector.historicalVariance(h);
        assertEquals(0, v[0], 1e-12);
        assertArrayEquals(new int[] {1, 2, 0}, SingleExampleSelector.rank(h));
    }

    @Test
    void tokenBroadcastAndSteps() {
        float[] b = TokenPolicyGradient.broadcast(List.of(new int[] {1, 2}, new int[] {3}), new float[] {5, 7});
        assertArrayEquals(new float[] {5, 5, 7}, b);
        assertArrayEquals(new int[] {2, 4, 5}, TokenPolicyGradient.stepEnds(new int[] {1, 9, 2, 9, 3}, 9));
    }

    @Test
    void primeOutcomeOnlyAdvantagesAreLeaveOneOut() {
        List<int[]> cs = List.of(new int[] {1}, new int[] {1});
        float[] a = ImplicitProcessReward.advantages(cs, new float[] {0, 0}, new float[] {1, 0}, 0);
        assertTrue(a[0] > 0 && a[1] < 0);
        assertEquals(-a[0], a[1], 1e-6);
    }
}
