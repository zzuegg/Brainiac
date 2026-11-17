package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.DeepConf;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.13: DeepConf offline and online against self-consistency and PRM best-of-N, 16 traces. */
class ConfidenceVotingTest {

    @Test
    void confidenceIsHighForPeakedDistributions() {
        float[] peaked = {10, 0, 0, 0, 0, 0};
        float[] flat = {0, 0, 0, 0, 0, 0};
        assertTrue(DeepConf.tokenConfidence(peaked, 3) > DeepConf.tokenConfidence(flat, 3));
        assertEquals(1.0, DeepConf.lowestGroupConfidence(new float[] {3, 1, 1, 3}, 2), 1e-9);
    }

    @Test
    void conservativeDeepConfMatchesVotingWithFewerTokens() {
        ConfidenceVoting.Result keep90 = null;
        for (double keep : new double[] {0.9, 0.5}) {
            ConfidenceVoting.Result r = ConfidenceVoting.run(new ChainTask(3), 40, 16, 4, keep, 2, 2, 1);
            System.out.printf("keep top %.0f%%: accuracy (tokens/question) self-consistency %.3f (%.1f), DeepConf-offline"
                    + " %.3f, DeepConf-online %.3f (%.1f), PRM best-of-16 %.3f (%.1f)%n", keep * 100, r.selfConsistency(),
                    r.selfConsistencyTokens(), r.offline(), r.online(), r.onlineTokens(), r.prmBestOfN(), r.prmTokens());
            keep90 = keep90 == null ? r : keep90;
        }
        assertTrue(keep90.offline() >= keep90.selfConsistency() - 0.05, "conservative filtering costs little accuracy");
        assertTrue(keep90.onlineTokens() < keep90.selfConsistencyTokens(), "early stopping saves tokens");
    }
}
