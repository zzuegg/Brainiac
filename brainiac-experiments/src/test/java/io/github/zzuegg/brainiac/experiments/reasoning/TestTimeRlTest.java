package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.TestTimeRl;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class TestTimeRlTest {

    @Test
    void majorityRewardsMarkTheVoteWinners() {
        float[] r = TestTimeRl.majorityRewards(Arrays.asList(3, 5, 3, null, 3, 5));
        assertArrayEquals(new float[] {1, 0, 1, 0, 1, 0}, r);
        assertArrayEquals(new float[] {0, 0}, TestTimeRl.majorityRewards(Arrays.<Integer>asList(null, null)));
    }

    /**
     * TTRL optimizes what it is told to (the groups agree more and more), but in this toy the
     * majority vote of 8 samples is <em>less</em> accurate than greedy decoding, so distilling
     * the vote pulls greedy accuracy down towards it: TTRL needs {@code maj@n > greedy}. The
     * test pins the mechanism (agreement rises) and prints the accuracies against the greedy
     * and self-consistency baselines.
     */
    @Test
    void ttrlRaisesAgreementWithTheMajority() {
        TestTimeScaling.TtrlResult r = TestTimeScaling.ttrl(new ChainTask(3), 60, 8, 40, 0.7, 3e-4f, 1);
        System.out.printf("TTRL greedy %.3f -> %.3f, sample %.3f -> %.3f, vote@8 %.3f -> %.3f, group agreement %.3f -> %.3f"
                + " (baselines: greedy %.3f, self-consistency %.3f)%n", r.greedy(), r.ttrlGreedy(), r.sample(),
                r.ttrlSample(), r.selfConsistency(), r.ttrlSelfConsistency(), r.agreementBefore(), r.agreementAfter(),
                r.greedy(), r.selfConsistency());
        assertTrue(r.agreementAfter() > r.agreementBefore(), r.toString());
        assertTrue(r.ttrlGreedy() > 0.2, r.toString());
    }
}
