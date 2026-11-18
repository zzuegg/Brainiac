package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.evolve.MapElites;
import io.github.zzuegg.brainiac.experimental.train.GFlowNet;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Sampling in proportion to a reward with a {@link GFlowNet}, against the alternatives:
 *
 * <ul>
 *   <li><b>Hypergrid</b> ({@link Hypergrid}, Bengio et al. 2021): an {@code H x H} grid walked from
 *       the corner by "+1 in x", "+1 in y" or stop; the reward has four separated high modes on a
 *       low floor. Compared by the L1 distance of the sampled terminal distribution to
 *       {@code R / Z}, and by how many of the four modes were found, against core {@link Ppo}
 *       (which maximizes reward and so collapses onto a mode), a Metropolis-Hastings random walk
 *       with the reward as target (unbiased, but it mixes slowly across the low-reward valleys)
 *       and {@link MapElites} (an archive of the best point per cell, which finds modes but
 *       defines no sampler).</li>
 *   <li><b>Sequences</b> ({@link Sequences}): strings of six letters over four symbols, built left
 *       to right, with reward {@code exp(-2 d)} for Hamming distance {@code d} to the nearest of
 *       four target strings, a small stand-in for molecule generation. GFlowNet against
 *       Metropolis-Hastings with single-letter mutations.</li>
 * </ul>
 * Malkin et al., <a href="https://arxiv.org/abs/2201.13259">arXiv:2201.13259</a>.
 */
public final class GFlowNetSampling {

    /**
     * One sampler's score.
     *
     * @param method the method
     * @param l1 {@code sum_x |p(x) - R(x)/Z|} of its empirical distribution (NaN for MAP-Elites)
     * @param modes distinct maximal-reward objects found
     * @param rewardCalls reward evaluations spent
     */
    public record Result(String method, double l1, int modes, long rewardCalls) {
    }

    private GFlowNetSampling() {
    }

    /** The hypergrid of Bengio et al. 2021 in two dimensions. */
    public static final class Hypergrid implements GFlowNet.Problem {
        private final int h;

        /**
         * @param h the side length
         */
        public Hypergrid(int h) {
            this.h = h;
        }

        /** @return the side length */
        public int side() {
            return h;
        }

        /**
         * The reward {@code 0.01 + 0.5 [all |x/(H-1) - 1/2| > 0.25] + 2 [all in (0.3, 0.4)]}.
         *
         * @param x a cell
         * @return the reward
         */
        public double reward(int[] x) {
            boolean outer = true;
            boolean band = true;
            for (int v : x) {
                double d = Math.abs((double) v / (h - 1) - 0.5);
                outer &= d > 0.25;
                band &= d > 0.3 && d < 0.4;
            }
            return 0.01 + (outer ? 0.5 : 0) + (band ? 2 : 0);
        }

        @Override
        public int[] initial() {
            return new int[2];
        }

        @Override
        public int actions() {
            return 3;
        }

        @Override
        public int stopAction() {
            return 2;
        }

        @Override
        public boolean[] legal(int[] s) {
            return new boolean[] {s[0] < h - 1, s[1] < h - 1, true};
        }

        @Override
        public int[] step(int[] s, int action) {
            int[] n = s.clone();
            n[action]++;
            return n;
        }

        @Override
        public int parents(int[] s) {
            return (s[0] > 0 ? 1 : 0) + (s[1] > 0 ? 1 : 0);
        }

        @Override
        public float[] features(int[] s) {
            float[] f = new float[2 * h];
            f[s[0]] = 1;
            f[h + s[1]] = 1;
            return f;
        }

        @Override
        public double logReward(int[] terminal) {
            return Math.log(reward(terminal));
        }

        /** @return {@code R(x) / Z} over all cells, row-major */
        public double[] target() {
            double[] p = new double[h * h];
            double z = 0;
            for (int i = 0; i < h * h; i++) {
                p[i] = reward(new int[] {i / h, i % h});
                z += p[i];
            }
            for (int i = 0; i < p.length; i++) {
                p[i] /= z;
            }
            return p;
        }

        /**
         * The same walk as an RL environment: reward {@code R(x)} on stopping, nothing before;
         * "+1" at the border is a no-op; episodes are cut after {@code 3H} steps.
         *
         * @return the environment
         */
        public Environment environment() {
            Hypergrid grid = this;
            return new Environment() {
                private int[] s = new int[2];
                private int steps;

                @Override
                public Space observationSpace() {
                    return Space.box(2 * h, 0, 1);
                }

                @Override
                public Space actionSpace() {
                    return Space.discrete(3);
                }

                @Override
                public float[] reset(SplittableRandom random) {
                    s = new int[2];
                    steps = 0;
                    return features(s);
                }

                @Override
                public Step step(float[] action) {
                    int a = Math.round(action[0]);
                    steps++;
                    if (a == 2) {
                        return new Step(features(s), (float) grid.reward(s), true, false);
                    }
                    if (s[a] < h - 1) {
                        s = grid.step(s, a);
                    }
                    return new Step(features(s), 0, false, steps >= 3 * h);
                }
            };
        }
    }

    /** Six-letter strings over four symbols with four target strings. */
    public static final class Sequences implements GFlowNet.Problem {
        private static final int LENGTH = 6;
        private static final int SYMBOLS = 4;
        private final int[][] targets;

        /**
         * @param seed picks the target strings
         */
        public Sequences(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            targets = new int[4][LENGTH];
            for (int[] t : targets) {
                for (int i = 0; i < LENGTH; i++) {
                    t[i] = r.nextInt(SYMBOLS);
                }
            }
        }

        private int distance(int[] x) {
            int best = LENGTH;
            for (int[] t : targets) {
                int d = 0;
                for (int i = 0; i < LENGTH; i++) {
                    d += t[i] == x[i] ? 0 : 1;
                }
                best = Math.min(best, d);
            }
            return best;
        }

        @Override
        public int[] initial() {
            return new int[0];
        }

        @Override
        public int actions() {
            return SYMBOLS + 1;
        }

        @Override
        public int stopAction() {
            return SYMBOLS;
        }

        @Override
        public boolean[] legal(int[] s) {
            boolean[] l = new boolean[SYMBOLS + 1];
            for (int a = 0; a < SYMBOLS; a++) {
                l[a] = s.length < LENGTH;
            }
            l[SYMBOLS] = s.length == LENGTH;
            return l;
        }

        @Override
        public int[] step(int[] s, int action) {
            int[] n = Arrays.copyOf(s, s.length + 1);
            n[s.length] = action;
            return n;
        }

        @Override
        public int parents(int[] s) {
            return 1;
        }

        @Override
        public float[] features(int[] s) {
            float[] f = new float[LENGTH * SYMBOLS + LENGTH + 1];
            for (int i = 0; i < s.length; i++) {
                f[i * SYMBOLS + s[i]] = 1;
            }
            f[LENGTH * SYMBOLS + s.length] = 1;
            return f;
        }

        @Override
        public double logReward(int[] terminal) {
            return -2.0 * distance(terminal);
        }

        private static int index(int[] x) {
            int k = 0;
            for (int v : x) {
                k = k * SYMBOLS + v;
            }
            return k;
        }

        /** @return {@code R(x)/Z} over all strings, by base-4 index */
        public double[] target() {
            int n = (int) Math.pow(SYMBOLS, LENGTH);
            double[] p = new double[n];
            double z = 0;
            for (int k = 0; k < n; k++) {
                int[] x = new int[LENGTH];
                int c = k;
                for (int i = LENGTH - 1; i >= 0; i--) {
                    x[i] = c % SYMBOLS;
                    c /= SYMBOLS;
                }
                p[k] = Math.exp(logReward(x));
                z += p[k];
            }
            for (int k = 0; k < n; k++) {
                p[k] /= z;
            }
            return p;
        }
    }

    private static double l1(double[] counts, double[] target) {
        double total = 0;
        for (double c : counts) {
            total += c;
        }
        double d = 0;
        for (int i = 0; i < target.length; i++) {
            d += Math.abs(counts[i] / total - target[i]);
        }
        return d;
    }

    /**
     * Trains a GFlowNet on the hypergrid and scores 2000 samples.
     *
     * @param grid the grid
     * @param steps training steps of 16 trajectories
     * @param seed the seed
     * @return the result
     */
    public static Result gflownet(Hypergrid grid, int steps, long seed) {
        GFlowNet g = GFlowNet.of(grid, 64, 1e-3f, seed).exploration(0.1);
        SplittableRandom r = new SplittableRandom(seed);
        Set<Integer> modes = new HashSet<>();
        for (int s = 0; s < steps; s++) {
            for (GFlowNet.Trajectory t : g.sample(16, 0.1, r)) {
                noteMode(grid, t.terminal(), modes);
            }
            g.trainStep(16, r);
        }
        double[] counts = new double[grid.side() * grid.side()];
        for (GFlowNet.Trajectory t : g.sample(2000, 0, r)) {
            counts[t.terminal()[0] * grid.side() + t.terminal()[1]]++;
        }
        return new Result("GFlowNet (TB)", l1(counts, grid.target()), modes.size(), g.rewardCalls());
    }

    private static void noteMode(Hypergrid grid, int[] x, Set<Integer> modes) {
        if (grid.reward(x) > 2) {
            modes.add(x[0] * grid.side() + x[1]);
        }
    }

    /**
     * Trains core PPO to maximize the reward and scores 2000 of its episodes.
     *
     * @param grid the grid
     * @param envSteps environment steps of training
     * @param seed the seed
     * @return the result
     */
    public static Result ppo(Hypergrid grid, long envSteps, long seed) {
        int in = 2 * grid.side();
        Ppo ppo = Ppo.of(Nets.mlp(in, 3, seed, 64, 64), Nets.mlp(in, 1, seed + 1, 64, 64), Space.discrete(3))
                .rolloutSteps(256).seed(seed);
        Set<Integer> modes = new HashSet<>();
        ppo.fit(grid.environment(), envSteps, RlListener.NONE);
        double[] counts = new double[grid.side() * grid.side()];
        for (int e = 0; e < 2000; e++) {
            int[] s = {0, 0};
            for (int t = 0; t < 3 * grid.side(); t++) {
                int a = Math.round(ppo.act(grid.features(s), true)[0]);
                if (a == 2) {
                    break;
                }
                if (s[a] < grid.side() - 1) {
                    s = grid.step(s, a);
                }
            }
            counts[s[0] * grid.side() + s[1]]++;
            noteMode(grid, s, modes);
        }
        return new Result("PPO (reward-max)", l1(counts, grid.target()), modes.size(), envSteps);
    }

    /**
     * A Metropolis-Hastings random walk with target {@code R}: propose a step of one cell in one
     * of four directions (rejected at the border), accept with {@code min(1, R(x')/R(x))}.
     *
     * @param grid the grid
     * @param evaluations reward evaluations (chain length)
     * @param seed the seed
     * @return the result over the chain after a 10% burn-in
     */
    public static Result mcmc(Hypergrid grid, long evaluations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] x = {0, 0};
        double rx = grid.reward(x);
        double[] counts = new double[grid.side() * grid.side()];
        Set<Integer> modes = new HashSet<>();
        for (long i = 0; i < evaluations; i++) {
            int[] y = x.clone();
            int d = r.nextInt(2);
            y[d] += r.nextBoolean() ? 1 : -1;
            if (y[d] >= 0 && y[d] < grid.side()) {
                double ry = grid.reward(y);
                if (r.nextDouble() < ry / rx) {
                    x = y;
                    rx = ry;
                }
            }
            noteMode(grid, x, modes);
            if (i >= evaluations / 10) {
                counts[x[0] * grid.side() + x[1]]++;
            }
        }
        return new Result("MCMC (MH)", l1(counts, grid.target()), modes.size(), evaluations);
    }

    /**
     * MAP-Elites over the continuous square with one cell per grid point.
     *
     * @param grid the grid
     * @param iterations evaluations after the initial population
     * @param seed the seed
     * @return modes found; no distribution
     */
    public static Result mapElites(Hypergrid grid, int iterations, long seed) {
        int h = grid.side();
        MapElites<double[]> me = MapElites.<double[]>of(
                r -> new double[] {r.nextDouble() * h, r.nextDouble() * h},
                MapElites.gaussian(1.0),
                x -> {
                    int[] c = {clamp((int) Math.floor(x[0]), h), clamp((int) Math.floor(x[1]), h)};
                    return new MapElites.Evaluation(grid.reward(c), new double[] {c[0] + 0.5, c[1] + 0.5});
                })
                .grid(new int[] {h, h}, new double[] {0, 0}, new double[] {h, h}).seed(seed).run(20, iterations);
        Set<Integer> modes = new HashSet<>();
        for (MapElites.Elite<double[]> e : me.elites()) {
            if (e.fitness() > 2) {
                modes.add((int) e.descriptor()[0] * h + (int) e.descriptor()[1]);
            }
        }
        return new Result("MAP-Elites", Double.NaN, modes.size(), me.evaluations());
    }

    private static int clamp(int v, int h) {
        return Math.max(0, Math.min(h - 1, v));
    }

    /**
     * GFlowNet and Metropolis-Hastings on the sequence task.
     *
     * @param problem the strings
     * @param steps GFlowNet steps of 16 trajectories (MCMC gets the same number of reward calls)
     * @param seed the seed
     * @return {GFlowNet, MCMC}
     */
    public static List<Result> sequences(Sequences problem, int steps, long seed) {
        double[] target = problem.target();
        GFlowNet g = GFlowNet.of(problem, 64, 1e-3f, seed).exploration(0.05);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            g.trainStep(16, r);
        }
        double[] counts = new double[target.length];
        Set<Integer> found = new HashSet<>();
        for (GFlowNet.Trajectory t : g.sample(4000, 0, r)) {
            int k = Sequences.index(t.terminal());
            counts[k]++;
            if (problem.distance(t.terminal()) == 0) {
                found.add(k);
            }
        }
        Result gfn = new Result("GFlowNet (TB)", l1(counts, target), found.size(), g.rewardCalls());
        int[] x = new int[Sequences.LENGTH];
        double lx = problem.logReward(x);
        double[] mc = new double[target.length];
        Set<Integer> mcFound = new HashSet<>();
        long evaluations = g.rewardCalls();
        for (long i = 0; i < evaluations; i++) {
            int[] y = x.clone();
            y[r.nextInt(y.length)] = r.nextInt(Sequences.SYMBOLS);
            double ly = problem.logReward(y);
            if (Math.log(r.nextDouble()) < ly - lx) {
                x = y;
                lx = ly;
            }
            if (problem.distance(x) == 0) {
                mcFound.add(Sequences.index(x));
            }
            if (i >= evaluations / 10) {
                mc[Sequences.index(x)]++;
            }
        }
        return List.of(gfn, new Result("MCMC (MH)", l1(mc, target), mcFound.size(), evaluations));
    }
}
