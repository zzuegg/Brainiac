package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class TestTimeComputeTest {

    @Test
    void majorityVoteIgnoresUnparsableAnswers() {
        assertEquals(3, TestTimeCompute.majorityVote(Arrays.asList(1, 3, null, 3, 2)));
        assertNull(TestTimeCompute.majorityVote(Arrays.asList(null, null)));
    }

    @Test
    void bestOfNAndCritiqueUseTheVerifier() {
        // a sampler that cycles through answers 0..4; the verifier likes 3
        int[] counter = {0};
        TestTimeCompute.Sampler sampler = (p, n, r) -> {
            int[][] out = new int[n][];
            for (int i = 0; i < n; i++) {
                out[i] = new int[] {counter[0]++ % 5};
            }
            return out;
        };
        TestTimeCompute.Scorer verifier = (p, c) -> c[0] == 3 ? 1 : 0;
        assertArrayEquals(new int[] {3}, TestTimeCompute.bestOfN(new int[0], sampler, 5, verifier,
                new SplittableRandom(1)));
        counter[0] = 0;
        TestTimeCompute.Critique c = TestTimeCompute.critiqueLoop(new int[0], sampler, verifier, 0.5, 10,
                new SplittableRandom(1));
        assertArrayEquals(new int[] {3}, c.completion());
        assertEquals(4, c.samples());
    }

    @Test
    void processRewardModelCutsSteps() {
        ProcessRewardModel prm = ProcessRewardModel.of(SequenceScorer.of(4, 4, 8, 1), 2, 3);
        List<int[]> cuts = prm.stepPrefixes(new int[] {0}, new int[] {1, 2, 1, 1, 3});
        assertEquals(2, cuts.size());
        assertArrayEquals(new int[] {0, 1, 2}, cuts.get(0));
        assertArrayEquals(new int[] {0, 1, 2, 1, 1, 3}, cuts.get(1));
        double[] s = prm.stepScores(new int[] {0}, new int[] {1, 2, 1, 1, 3});
        assertEquals(2, s.length);
        double min = prm.score(new int[] {0}, new int[] {1, 2, 1, 1, 3}, ProcessRewardModel.Aggregate.MIN);
        assertEquals(Math.min(s[0], s[1]), min, 1e-12);
    }

    @Test
    void sequenceScorerLearnsALabel() {
        SequenceScorer s = SequenceScorer.of(3, 3, 16, 1);
        List<int[]> seqs = List.of(new int[] {0, 1}, new int[] {1, 0}, new int[] {2, 2}, new int[] {0, 2});
        s.fit(seqs, new float[] {1, 0, 0, 1}, 200);
        assertTrue(s.score(seqs.get(0)) > 0.9 && s.score(seqs.get(1)) < 0.1);
    }
}
