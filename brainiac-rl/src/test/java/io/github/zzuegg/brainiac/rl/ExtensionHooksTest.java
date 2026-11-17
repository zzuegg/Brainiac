package io.github.zzuegg.brainiac.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Hooks for agents and learners written outside the module. */
class ExtensionHooksTest {

    /** Counts episode starts; acts at random. */
    static final class Counting implements Agent {
        int starts;
        int actsSinceStart;
        final SplittableRandom random = new SplittableRandom(1);

        @Override
        public float[] act(float[] observation, boolean explore) {
            actsSinceStart++;
            return new float[] {random.nextInt(4)};
        }

        @Override
        public void startEpisode() {
            starts++;
            actsSinceStart = 0;
        }
    }

    @Test
    void evaluationStartsEachEpisode() {
        Counting agent = new Counting();
        Rl.evaluate(agent, GridWorld.cliff(), 3, 1);
        assertEquals(3, agent.starts);
        Rl.watch(agent, GridWorld.cliff(), 1, 5);
        assertEquals(4, agent.starts);
    }

    @Test
    void trackerRecordsEpisodesOfACustomLearner() {
        int[] seen = new int[1];
        Tracker tracker = new Tracker((episode, result) -> seen[0]++);
        Environment env = GridWorld.cliff();
        Counting agent = new Counting();
        float[] obs = env.reset(new SplittableRandom(2));
        while (tracker.result().episodes().size() < 2) {
            Environment.Step s = env.step(agent.act(obs, true));
            obs = tracker.step(s) ? env.reset(new SplittableRandom(3)) : s.observation();
        }
        assertEquals(2, seen[0]);
        assertEquals(tracker.steps(), tracker.result().totalSteps());
    }
}
