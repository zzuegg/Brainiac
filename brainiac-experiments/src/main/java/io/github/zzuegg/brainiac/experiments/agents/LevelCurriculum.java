package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.LevelReplay;
import io.github.zzuegg.brainiac.experimental.rl.ReplayedLevels;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.GridWorld;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Prioritized Level Replay against uniform level sampling on procedurally generated mazes,
 * with a tabular Q-learner and the same number of episodes. PLR scores each level by the mean
 * absolute TD error of its last episode, so it stops replaying mazes that are solved or
 * unsolvable (both have no error left) and spends the budget where the learner is still
 * improving.
 */
public final class LevelCurriculum {

    private static final int SIZE = 6;

    /**
     * @param plrSolved solvable levels the PLR-trained learner solves greedily
     * @param uniformSolved those the uniformly trained learner solves
     * @param solvable solvable levels in the set
     * @param levels levels in the set
     */
    public record Result(int plrSolved, int uniformSolved, int solvable, int levels) {
    }

    private LevelCurriculum() {
    }

    /**
     * Level {@code i}: a random 6x6 maze, start in the top-left corner, goal somewhere else.
     *
     * @param i the level id
     * @return the maze rows
     */
    public static String[] maze(int i) {
        SplittableRandom r = new SplittableRandom(1000L + i);
        char[][] g = new char[SIZE][SIZE];
        for (char[] row : g) {
            for (int c = 0; c < SIZE; c++) {
                row[c] = r.nextDouble() < 0.3 ? '#' : '.';
            }
        }
        g[0][0] = 'S';
        int goal = 1 + r.nextInt(SIZE * SIZE - 1);
        g[goal / SIZE][goal % SIZE] = 'G';
        String[] rows = new String[SIZE];
        for (int k = 0; k < SIZE; k++) {
            rows[k] = new String(g[k]);
        }
        return rows;
    }

    static boolean solvable(String[] m) {
        boolean[] seen = new boolean[SIZE * SIZE];
        ArrayDeque<Integer> q = new ArrayDeque<>();
        q.add(0);
        seen[0] = true;
        int[][] moves = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        while (!q.isEmpty()) {
            int c = q.poll();
            if (m[c / SIZE].charAt(c % SIZE) == 'G') {
                return true;
            }
            for (int[] mv : moves) {
                int r = c / SIZE + mv[0];
                int k = c % SIZE + mv[1];
                if (r >= 0 && r < SIZE && k >= 0 && k < SIZE && m[r].charAt(k) != '#' && !seen[r * SIZE + k]) {
                    seen[r * SIZE + k] = true;
                    q.add(r * SIZE + k);
                }
            }
        }
        return false;
    }

    private static Environment level(int i) {
        return GridWorld.of(maze(i)).stepReward(-0.02f).maxSteps(30);
    }

    /**
     * @param levels levels in the training set
     * @param episodes training episodes
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int levels, int episodes, long seed) {
        return run(levels, episodes, 0.2, 0.05, seed);
    }

    static Result run(int levels, int episodes, double beta, double rho, long seed) {
        int solvable = 0;
        for (int i = 0; i < levels; i++) {
            solvable += solvable(maze(i)) ? 1 : 0;
        }
        LevelReplay plr = LevelReplay.of(levels, seed).replayProbability(0.5).temperature(beta).staleness(rho);
        LevelReplay uniform = LevelReplay.of(levels, seed).replayProbability(0.5).temperature(1e9).staleness(0);
        return new Result(train(plr, levels, episodes, seed), train(uniform, levels, episodes, seed), solvable, levels);
    }

    private static int train(LevelReplay sampler, int levels, int episodes, long seed) {
        ReplayedLevels env = ReplayedLevels.of(LevelCurriculum::level, sampler);
        Map<Integer, float[]> q = new HashMap<>();
        SplittableRandom random = new SplittableRandom(seed);
        int cells = SIZE * SIZE;
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset(random.split());
            int base = env.currentLevel() * cells;
            int s = base + argmax(obs);
            double error = 0;
            int steps = 0;
            boolean done = false;
            while (!done) {
                float[] qs = q.computeIfAbsent(s, k -> new float[4]);
                int a = random.nextDouble() < 0.2 ? random.nextInt(4) : argmax(qs);
                Environment.Step st = env.step(new float[] {a});
                int s2 = base + argmax(st.observation());
                float[] q2 = q.computeIfAbsent(s2, k -> new float[4]);
                double target = st.reward() + (st.terminated() ? 0 : 0.95 * q2[argmax(q2)]);
                double delta = target - qs[a];
                qs[a] += (float) (0.5 * delta);
                error += Math.abs(delta);
                steps++;
                s = s2;
                done = st.done();
            }
            env.score(error / steps);
        }
        int solved = 0;
        for (int i = 0; i < levels; i++) {
            Environment lvl = level(i);
            float[] obs = lvl.reset(random.split());
            boolean done = false;
            while (!done) {
                float[] qs = q.getOrDefault(i * cells + argmax(obs), new float[4]);
                Environment.Step st = lvl.step(new float[] {argmax(qs)});
                if (st.terminated() && st.reward() > 0) {
                    solved++;
                }
                obs = st.observation();
                done = st.done();
            }
        }
        return solved;
    }

    private static int argmax(float[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[b]) {
                b = i;
            }
        }
        return b;
    }
}
