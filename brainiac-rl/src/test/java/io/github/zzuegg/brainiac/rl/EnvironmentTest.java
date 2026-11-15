package io.github.zzuegg.brainiac.rl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Environments, spaces, schedules and replay buffers. */
class EnvironmentTest {

    private static int episodeLength(Environment env, int action, long seed) {
        env.reset(new SplittableRandom(seed));
        int steps = 0;
        Environment.Step s;
        do {
            s = env.step(action);
            steps++;
        } while (!s.done());
        return steps;
    }

    @Test
    void cartPoleFallsWhenPushedOneWayAndIsCutOffAtTheLimit() {
        CartPole env = new CartPole();
        int steps = episodeLength(env, 1, 1);
        assertTrue(steps < 20, "always pushing right falls quickly, took " + steps);
        Environment.Step s = env.reset(new SplittableRandom(1)).length == 4 ? env.step(0) : null;
        assertEquals(1f, s.reward());
        assertThrows(IllegalArgumentException.class, () -> env.step(2));
        CartPole shortEpisodes = new CartPole(5);
        shortEpisodes.reset(new SplittableRandom(2));
        for (int i = 0; i < 4; i++) {
            assertFalse(shortEpisodes.step(i % 2).done());
        }
        Environment.Step last = shortEpisodes.step(1);
        assertTrue(last.truncated() && !last.terminated());
        assertThrows(IllegalStateException.class, () -> shortEpisodes.step(0));
    }

    @Test
    void sameSeedGivesTheSameEpisode() {
        float[] a = new CartPole().reset(new SplittableRandom(7));
        float[] b = new CartPole().reset(new SplittableRandom(7));
        assertArrayEquals(a, b);
    }

    @Test
    void mountainCarCannotDriveStraightUp() {
        assertEquals(200, episodeLength(new MountainCar(), 2, 1));
        // swinging: push in the direction of motion
        MountainCar env = new MountainCar(1000);
        float[] obs = env.reset(new SplittableRandom(1));
        Environment.Step s;
        int steps = 0;
        do {
            s = env.step(obs[1] < 0 ? 0 : 2);
            obs = s.observation();
            steps++;
        } while (!s.done());
        assertTrue(s.terminated(), "momentum reaches the flag");
        assertTrue(steps < 200, "in " + steps + " steps");
        MountainCar shaped = new MountainCar().shaped(true);
        shaped.reset(new SplittableRandom(1));
        assertTrue(shaped.toString().contains("shaped"));
    }

    @Test
    void pendulumHasContinuousActionsAndRewardsUprightness() {
        Pendulum env = new Pendulum();
        assertTrue(env.actionSpace() instanceof Space.Box b && b.size() == 1);
        env.reset(new SplittableRandom(3));
        Environment.Step s = env.step(new float[] {5});    // clipped to 2
        assertTrue(s.reward() <= 0);
        assertEquals(200, episodeLength(env, 0, 4));
    }

    @Test
    void gridWorldMovesBumpsAndEnds() {
        GridWorld g = GridWorld.of("S.#", "..G");
        float[] obs = g.reset(new SplittableRandom(1));
        assertEquals(6, obs.length);
        assertEquals(1, obs[0]);
        g.step(1);                                // right
        Environment.Step s = g.step(1);           // wall: stays
        assertEquals(1, s.observation()[1]);
        s = g.step(2);                            // down
        assertFalse(s.done());
        s = g.step(1);                            // right onto the goal
        assertTrue(s.terminated());
        assertEquals(1 - 0.01f, s.reward(), 1e-6);
        GridWorld cliff = GridWorld.cliff();
        cliff.reset(new SplittableRandom(1));
        s = cliff.step(1);
        assertEquals(-100, s.reward());
        assertEquals(1, s.observation()[36], "back at the start");
        GridWorld coords = GridWorld.of("S.", ".G").coordinates(true);
        assertArrayEquals(new float[] {0, 0}, coords.reset(new SplittableRandom(1)));
        assertThrows(IllegalArgumentException.class, () -> GridWorld.of("S.", "..."));
        assertThrows(IllegalArgumentException.class, () -> GridWorld.of("..", ".G"));
        assertTrue(g.render().contains("@"));
    }

    @Test
    void timeLimitTruncates() {
        Environment env = GridWorld.of("S...G").maxSteps(1000).limit(3);
        assertEquals(3, episodeLength(env, 3, 1));
    }

    @Test
    void banditPaysAroundEachArmsMean() {
        Bandit b = new Bandit(new double[] {0, 2, 1}, 0.1);
        assertEquals(1, b.bestArm());
        double sum = 0;
        SplittableRandom r = new SplittableRandom(1);
        for (int i = 0; i < 200; i++) {
            b.reset(r);
            sum += b.step(1).reward();
        }
        assertEquals(2, sum / 200, 0.05);
    }

    @Test
    void spacesAndSchedules() {
        Space.Box box = Space.box(new float[] {-1, 0}, new float[] {1, 5});
        assertArrayEquals(new float[] {-1, 5}, box.clip(new float[] {-3, 9}));
        float[] x = box.sample(new SplittableRandom(1));
        assertTrue(x[0] >= -1 && x[0] <= 1 && x[1] >= 0 && x[1] <= 5);
        assertEquals(3, Space.discrete(3).n());
        assertThrows(IllegalArgumentException.class, () -> Space.discrete(0));
        Schedule s = Schedule.linear(1, 0.1, 10);
        assertEquals(1, s.at(0), 1e-9);
        assertEquals(0.55, s.at(5), 1e-9);
        assertEquals(0.1, s.at(100), 1e-9);
        assertEquals(0.5, Schedule.exponential(1, 0.5, 0.5).at(3), 1e-9);
        assertEquals(0.25, Schedule.exponential(1, 0.1, 0.5).at(2), 1e-9);
    }

    @Test
    void replayBufferWrapsAround() {
        ReplayBuffer b = new ReplayBuffer(3, 2, 1);
        for (int i = 0; i < 5; i++) {
            b.add(new Transition(new float[] {i, i}, new float[] {i % 2}, i, new float[] {i + 1, i + 1}, i == 4,
                    0.9f));
        }
        assertEquals(3, b.size());
        ReplayBuffer.Batch batch = b.sample(50, new SplittableRandom(1), io.github.zzuegg.brainiac.tensor.Device.cpu());
        float[] rewards = batch.rewards().toArray();
        for (float r : rewards) {
            assertTrue(r >= 2, "the oldest transitions were overwritten");
        }
        assertEquals(io.github.zzuegg.brainiac.tensor.Shape.of(50, 2), batch.observations().shape());
    }

    @Test
    void prioritizedReplayFavoursLargeErrors() {
        PrioritizedReplayBuffer b = new PrioritizedReplayBuffer(8, 1, 1, 0.5);
        for (int i = 0; i < 8; i++) {
            b.add(new Transition(new float[] {i}, new float[] {0}, 0, new float[] {i}, false, 1));
        }
        int[] all = {0, 1, 2, 3, 4, 5, 6, 7};
        float[] errors = {0.01f, 0.01f, 0.01f, 10f, 0.01f, 0.01f, 0.01f, 0.01f};
        b.updatePriorities(all, errors);
        ReplayBuffer.Batch batch = b.sample(100, new SplittableRandom(2), io.github.zzuegg.brainiac.tensor.Device.cpu());
        int hits = 0;
        float[] w = batch.weights().toArray();
        for (int i = 0; i < 100; i++) {
            if (batch.indices()[i] == 3) {
                hits++;
                assertTrue(w[i] < 1, "frequent samples get smaller weights");
            }
        }
        assertTrue(hits > 60 && hits < 100, "slot 3 sampled " + hits + " times");
        assertEquals(10, b.priority(3), 1e-3);
    }
}
