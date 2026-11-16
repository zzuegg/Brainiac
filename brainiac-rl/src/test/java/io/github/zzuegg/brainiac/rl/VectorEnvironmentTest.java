package io.github.zzuegg.brainiac.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Stepping several environments together, and training on them. */
class VectorEnvironmentTest {

    @Test
    void entriesStepAndResetOnTheirOwn() {
        VectorEnvironment v = VectorEnvironment.of(List.of(new CartPole(3), new CartPole(5)));
        assertEquals(2, v.count());
        assertEquals(2, v.reset(new SplittableRandom(1)).length);
        for (int t = 0; t < 2; t++) {
            v.step(new float[][] {{0}, {1}});
        }
        Environment.Step[] s = v.step(new float[][] {{1}, {0}});
        assertTrue(s[0].truncated(), "the first is cut off after 3 steps");
        v.reset(0, new SplittableRandom(2));
        s = v.step(new float[][] {{0}, {1}});
        assertTrue(!s[0].done(), "and starts again");
    }

    @Test
    void concatenatedVectorsRouteActionsAndResets() {
        VectorEnvironment a = VectorEnvironment.of(List.of(new CartPole(2)));
        VectorEnvironment b = VectorEnvironment.of(List.of(new CartPole(4), new CartPole(4)));
        VectorEnvironment all = VectorEnvironment.concat(List.of(a, b));
        assertEquals(3, all.count());
        all.reset(new SplittableRandom(1));
        all.step(new float[][] {{0}, {0}, {0}});
        Environment.Step[] s = all.step(new float[][] {{0}, {0}, {0}});
        assertTrue(s[0].done() && !s[1].done() && !s[2].done());
        all.reset(0, new SplittableRandom(3));
        assertTrue(!all.step(new float[][] {{0}, {0}, {0}})[0].done());
    }

    @Test
    void episodesOfSideBySideEnvironmentsAreCountedSeparately() {
        Ppo ppo = Ppo.of(Nets.mlp(4, 2, 1, 16), Nets.mlp(4, 1, 2, 16), Space.discrete(2)).rolloutSteps(16)
                .seed(1);
        RlResult r = ppo.fit(VectorEnvironment.of(List.of(new CartPole(3), new CartPole(7))), 400, RlListener.NONE);
        for (Episode e : r.episodes()) {
            assertTrue(e.steps() <= 7, "an episode of " + e.steps() + " steps mixes the two environments");
        }
        assertTrue(r.episodes().stream().anyMatch(e -> e.steps() == 3), "the short one is cut off at 3");
    }

    @Test
    void theAdaptiveLearningRateFollowsTheKl() {
        Ppo calm = Ppo.of(Nets.mlp(4, 2, 1, 16), Nets.mlp(4, 1, 2, 16), Space.discrete(2)).rolloutSteps(32)
                .learningRate(1e-6f).adaptiveLearningRate(0.01f).seed(1);
        calm.fit(new CartPole(), 32 * 6, RlListener.NONE);
        assertTrue(calm.learningRateScale() > 1, "tiny steps raise the rate: " + calm.learningRateScale());
        Ppo wild = Ppo.of(Nets.mlp(4, 2, 1, 16), Nets.mlp(4, 1, 2, 16), Space.discrete(2)).rolloutSteps(32)
                .learningRate(3e-3f).adaptiveLearningRate(1e-7f).seed(1);
        wild.fit(new CartPole(), 32 * 6, RlListener.NONE);
        assertTrue(wild.learningRateScale() < 1, "steps far above the target lower it: " + wild.learningRateScale());
    }
}
