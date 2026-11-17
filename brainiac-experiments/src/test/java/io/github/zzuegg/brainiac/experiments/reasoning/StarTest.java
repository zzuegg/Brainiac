package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** STaR with rationalization against rejection-sampling fine-tuning (STaR without it). */
class StarTest {

    @Test
    void rationalizationBeatsRejectionSamplingFineTuning() {
        ChainTask task = new ChainTask(3);
        SelfImprovement.Result star = SelfImprovement.star(task, 12, 3, true, 1);
        SelfImprovement.Result rft = SelfImprovement.star(task, 12, 3, false, 1);
        System.out.printf("STaR greedy accuracy %.3f -> %.3f vs RFT (no rationalization) %.3f -> %.3f, 3 rounds%n",
                star.before(), star.after(), rft.before(), rft.after());
        assertTrue(star.after() > star.before(), star.toString());
        assertTrue(star.after() >= rft.after(), star + " vs " + rft);
    }
}
