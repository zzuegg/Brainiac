package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TestTimeScalingTest {

    @Test
    void moreInferenceComputeBeatsOneSample() {
        TestTimeScaling.Result r = TestTimeScaling.run(new ChainTask(3), 40, 10, 1);
        System.out.println(r);
        double base = r.singleSample();
        assertTrue(base < 0.8, r.toString());
        assertTrue(r.bestOfN() > base + 0.1, "best-of-N " + r);
        assertTrue(r.selfConsistency() >= base, "self-consistency " + r);
        assertTrue(r.weightedVote() > base + 0.1, "weighted vote " + r);
        assertTrue(r.critique() > base + 0.1, "critique loop " + r);
        assertTrue(r.critiqueSamples() < 10, "the critique loop stops early on easy questions " + r);
        assertTrue(r.prmBestOfN() > base + 0.1, "PRM best-of-N " + r);
        assertTrue(r.beamSearch() > base + 0.1, "beam search " + r);
        assertTrue(r.depthFirst() > base + 0.1, "depth-first search " + r);
    }
}
