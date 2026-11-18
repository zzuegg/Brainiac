package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.OperationsSims.VectorMdp;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.IntUnaryOperator;

/**
 * Multi-objective RL on deterministic vector-reward MDPs ({@link VectorMdp}), with the
 * coverage metrics of the field:
 *
 * <ul>
 *   <li>{@link #scalarizedQ}: linear scalarisation, one tabular Q-learner per preference weight
 *       (Roijers et al., JAIR 2013): it can only reach points on the convex hull of the front;</li>
 *   <li>{@link #paretoQ}: Pareto Q-learning (Van Moffaert &amp; Nowé, JMLR 2014): learned average
 *       immediate rewards and successor states, and set-valued backups
 *       {@code Q(s, a) = R(s, a) + gamma ND(union_a' Q(s', a'))} that keep every non-dominated
 *       return vector, so the whole front (concave parts included) is found;</li>
 *   <li>{@link Envelope}: envelope Q-learning (Yang, Sun &amp; Narasimhan, NeurIPS 2019,
 *       <a href="https://arxiv.org/abs/1908.08342">arXiv:1908.08342</a>): one preference-conditioned
 *       network {@code Q(s, a, w)} with vector outputs, trained towards
 *       {@code r + gamma Q(s', a*, w*)} with {@code (a*, w*) = argmax_(a', w') w . Q(s', a', w')} over
 *       actions and a batch of sampled preferences (the envelope), so it generalises across
 *       preferences.</li>
 * </ul>
 *
 * <p>{@link #hypervolume} (two objectives) and {@link #nondominated} score coverage sets;
 * {@link #scalarised} exposes a preference-weighted MDP as a core {@link Environment} for
 * scalarised core DQN baselines.
 */
public final class MultiObjectiveRl {

    private MultiObjectiveRl() {
    }

    /**
     * Linear-scalarisation Q-learning for one preference, then the greedy policy's return vector.
     *
     * @param mdp the MDP
     * @param weights the preference
     * @param episodes training episodes
     * @param seed the seed
     * @return the greedy policy's return vector (undiscounted)
     */
    public static double[] scalarizedQ(VectorMdp mdp, double[] weights, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int na = mdp.actions();
        double[][] q = new double[mdp.states()][na];
        for (int ep = 0; ep < episodes; ep++) {
            double eps = Math.max(0.05, 1 - ep / (0.7 * episodes));
            int s = mdp.start();
            for (int t = 0; t < mdp.maxSteps() && !mdp.terminal()[s]; t++) {
                int a = r.nextDouble() < eps ? r.nextInt(na) : argmax(q[s], r);
                int s2 = mdp.next()[s][a];
                double rew = dot(weights, mdp.reward()[s][a]);
                double target = rew + (mdp.terminal()[s2] ? 0 : max(q[s2]));
                q[s][a] += 0.2 * (target - q[s][a]);
                s = s2;
            }
        }
        return rollout(mdp, s -> argmax(q[s], null));
    }

    /**
     * Pareto Q-learning: random exploration learns the (deterministic) model, then
     * {@code maxSteps} rounds of set-valued backups; returns the non-dominated return vectors
     * at the start state.
     *
     * @param mdp the MDP
     * @param episodes exploration episodes
     * @param seed the seed
     * @return the coverage set found
     */
    public static List<double[]> paretoQ(VectorMdp mdp, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = mdp.states();
        int na = mdp.actions();
        int m = mdp.objectives();
        double[][][] avgReward = new double[n][na][m];
        int[][] count = new int[n][na];
        int[][] next = new int[n][na];
        for (int ep = 0; ep < episodes; ep++) {
            int s = mdp.start();
            for (int t = 0; t < mdp.maxSteps() && !mdp.terminal()[s]; t++) {
                int a = r.nextInt(na);
                int s2 = mdp.next()[s][a];
                count[s][a]++;
                for (int k = 0; k < m; k++) {
                    avgReward[s][a][k] += (mdp.reward()[s][a][k] - avgReward[s][a][k]) / count[s][a];
                }
                next[s][a] = s2;
                s = s2;
            }
        }
        List<List<double[]>> v = new ArrayList<>();
        for (int s = 0; s < n; s++) {
            List<double[]> zero = new ArrayList<>();
            zero.add(new double[m]);
            v.add(zero);
        }
        for (int sweep = 0; sweep < mdp.maxSteps(); sweep++) {
            List<List<double[]>> nv = new ArrayList<>();
            for (int s = 0; s < n; s++) {
                List<double[]> union = new ArrayList<>();
                if (!mdp.terminal()[s]) {
                    for (int a = 0; a < na; a++) {
                        if (count[s][a] == 0) {
                            continue;
                        }
                        for (double[] w : v.get(next[s][a])) {
                            double[] x = new double[m];
                            for (int k = 0; k < m; k++) {
                                x[k] = avgReward[s][a][k] + w[k];
                            }
                            union.add(x);
                        }
                    }
                }
                if (union.isEmpty()) {
                    union.add(new double[m]);
                }
                nv.add(nondominated(union));
            }
            v = nv;
        }
        return v.get(mdp.start());
    }

    /**
     * The return vector of a deterministic policy, capped at the MDP's episode length.
     *
     * @param mdp the MDP
     * @param policy the action per state
     * @return the undiscounted return vector
     */
    public static double[] rollout(VectorMdp mdp, IntUnaryOperator policy) {
        double[] ret = new double[mdp.objectives()];
        int s = mdp.start();
        for (int t = 0; t < mdp.maxSteps() && !mdp.terminal()[s]; t++) {
            int a = policy.applyAsInt(s);
            for (int k = 0; k < ret.length; k++) {
                ret[k] += mdp.reward()[s][a][k];
            }
            s = mdp.next()[s][a];
        }
        return ret;
    }

    /**
     * @param points vectors (maximisation in every objective)
     * @return the non-dominated ones, duplicates removed, sorted by the first objective
     */
    public static List<double[]> nondominated(List<double[]> points) {
        List<double[]> out = new ArrayList<>();
        for (double[] p : points) {
            boolean dominated = false;
            for (double[] q : points) {
                if (q != p && dominates(q, p)) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated && out.stream().noneMatch(o -> same(o, p))) {
                out.add(p);
            }
        }
        out.sort(Comparator.comparingDouble(x -> x[0]));
        return out;
    }

    private static boolean dominates(double[] a, double[] b) {
        boolean strict = false;
        for (int k = 0; k < a.length; k++) {
            if (a[k] < b[k] - 1e-9) {
                return false;
            }
            strict |= a[k] > b[k] + 1e-9;
        }
        return strict;
    }

    private static boolean same(double[] a, double[] b) {
        for (int k = 0; k < a.length; k++) {
            if (Math.abs(a[k] - b[k]) > 1e-9) {
                return false;
            }
        }
        return true;
    }

    /**
     * The two-objective hypervolume dominated by a point set and bounded by a reference point.
     *
     * @param points the points
     * @param reference a point dominated by all of them
     * @return the area
     */
    public static double hypervolume(List<double[]> points, double[] reference) {
        List<double[]> front = new ArrayList<>(nondominated(points));
        front.sort((x, y) -> Double.compare(y[0], x[0]));
        double area = 0;
        double lastY = reference[1];
        for (double[] p : front) {
            if (p[0] <= reference[0] || p[1] <= lastY) {
                continue;
            }
            area += (p[0] - reference[0]) * (p[1] - lastY);
            lastY = p[1];
        }
        return area;
    }

    /**
     * The MDP scalarised by a preference, as a core environment with one-hot observations.
     *
     * @param mdp the MDP
     * @param weights the preference
     * @return the environment
     */
    public static Environment scalarised(VectorMdp mdp, double[] weights) {
        return new Environment() {
            private int s;
            private int t;

            @Override
            public Space observationSpace() {
                return Space.box(mdp.states(), 0, 1);
            }

            @Override
            public Space actionSpace() {
                return Space.discrete(mdp.actions());
            }

            private float[] obs() {
                float[] o = new float[mdp.states()];
                o[s] = 1;
                return o;
            }

            @Override
            public float[] reset(SplittableRandom random) {
                s = mdp.start();
                t = 0;
                return obs();
            }

            @Override
            public Step step(float[] action) {
                int a = Math.round(action[0]);
                double rew = dot(weights, mdp.reward()[s][a]);
                s = mdp.next()[s][a];
                t++;
                return new Step(obs(), (float) rew, mdp.terminal()[s], t >= mdp.maxSteps());
            }
        };
    }

    /**
     * Envelope Q-learning with a preference-conditioned MLP (input: the one-hot state and the
     * preference; output: one Q-vector per action). The envelope target picks the action and
     * sampled preference with the online network and evaluates it with a target network synced
     * every 100 updates (double-Q style), clipped to the range the rewards allow; one update every
     * four environment steps.
     */
    public static final class Envelope {
        private final int in;
        private final VectorMdp mdp;
        private final double[] scale;
        private final Model net;
        private final Model target;
        private final Optimizer opt;
        private final SplittableRandom random;
        private final int na;
        private final int m;
        private final List<double[]> replay = new ArrayList<>();
        private int preferenceSamples = 4;
        private int updates;
        private final double[] low;
        private final double[] high;
        private int batch = 32;
        private double gamma = 0.98;

        private Envelope(VectorMdp mdp, double[] scale, long seed) {
            this.mdp = mdp;
            this.scale = scale.clone();
            this.na = mdp.actions();
            this.m = mdp.objectives();
            this.in = mdp.states();
            this.net = Nets.mlp(in + m, na * m, seed, 64, 64);
            this.target = Nets.mlp(in + m, na * m, seed, 64, 64);
            this.opt = Optimizer.adam(1e-3f);
            this.random = new SplittableRandom(seed);
            this.low = new double[m];
            this.high = new double[m];
            for (double[][] perState : mdp.reward()) {
                for (double[] r : perState) {
                    for (int k = 0; k < m; k++) {
                        low[k] = Math.min(low[k], mdp.maxSteps() * r[k] / scale[k]);
                        high[k] = Math.max(high[k], mdp.maxSteps() * r[k] / scale[k]);
                    }
                }
            }
        }

        /**
         * @param mdp the MDP
         * @param rewardScale divides each objective (to bring them to unit scale)
         * @param seed the seed
         * @return the learner
         */
        public static Envelope of(VectorMdp mdp, double[] rewardScale, long seed) {
            return new Envelope(mdp, rewardScale, seed);
        }

        /** @param g the discount @return this learner */
        public Envelope discount(double g) {
            this.gamma = g;
            return this;
        }

        /**
         * @param s a state
         * @param w a preference
         * @return the predicted Q-vectors, action-major ({@code [a * objectives + k]})
         */
        public float[] qValues(int s, double[] w) {
            return predict(net, new float[][] {input(s, w)})[0];
        }

        /** @param n preferences sampled for the envelope @return this learner */
        public Envelope preferenceSamples(int n) {
            this.preferenceSamples = n;
            return this;
        }

        private float[] input(int s, double[] w) {
            float[] x = new float[in + m];
            x[s] = 1;
            for (int k = 0; k < m; k++) {
                x[in + k] = (float) w[k];
            }
            return x;
        }

        private double[] randomPreference() {
            double[] w = new double[m];
            double sum = 0;
            for (int k = 0; k < m; k++) {
                w[k] = -Math.log(1 - random.nextDouble());
                sum += w[k];
            }
            for (int k = 0; k < m; k++) {
                w[k] /= sum;
            }
            return w;
        }

        private float[][] predict(Model model, float[][] rows) {
            int b = rows.length;
            float[] flat = new float[b * (in + m)];
            for (int i = 0; i < b; i++) {
                System.arraycopy(rows[i], 0, flat, i * (in + m), in + m);
            }
            float[] out = model.forward(Var.of(Tensor.of(Shape.of(b, in + m), flat)), false).value().toArray();
            float[][] q = new float[b][];
            for (int i = 0; i < b; i++) {
                q[i] = java.util.Arrays.copyOfRange(out, i * na * m, (i + 1) * na * m);
            }
            return q;
        }

        /**
         * @param s a state
         * @param w a preference
         * @return the greedy action under {@code w . Q(s, ., w)}
         */
        public int act(int s, double[] w) {
            float[] q = predict(net, new float[][] {input(s, w)})[0];
            int best = 0;
            double bv = Double.NEGATIVE_INFINITY;
            for (int a = 0; a < na; a++) {
                double v = 0;
                for (int k = 0; k < m; k++) {
                    v += w[k] * q[a * m + k];
                }
                if (v > bv) {
                    bv = v;
                    best = a;
                }
            }
            return best;
        }

        /**
         * Trains for a number of environment steps (one preference per episode, epsilon-greedy).
         *
         * @param steps environment steps
         * @return this learner
         */
        public Envelope train(int steps) {
            int s = mdp.start();
            int t = 0;
            double[] w = randomPreference();
            for (int step = 0; step < steps; step++) {
                double eps = Math.max(0.1, 1 - step / (0.5 * steps));
                int a = random.nextDouble() < eps ? random.nextInt(na) : act(s, w);
                int s2 = mdp.next()[s][a];
                double[] rec = new double[3 + m];
                rec[0] = s;
                rec[1] = a;
                rec[2] = s2;
                for (int k = 0; k < m; k++) {
                    rec[3 + k] = mdp.reward()[s][a][k] / scale[k];
                }
                replay.add(rec);
                t++;
                s = s2;
                if (mdp.terminal()[s] || t >= mdp.maxSteps()) {
                    s = mdp.start();
                    t = 0;
                    w = randomPreference();
                }
                if (replay.size() >= batch && step % 4 == 0) {
                    update();
                }
            }
            return this;
        }

        private void update() {
            int np = preferenceSamples;
            double[][] prefs = new double[np][];
            for (int j = 0; j < np; j++) {
                prefs[j] = randomPreference();
            }
            double[][] batchRec = new double[batch][];
            for (int i = 0; i < batch; i++) {
                batchRec[i] = replay.get(random.nextInt(replay.size()));
            }
            // targets for every (sample, preference) pair
            float[][] nextRows = new float[batch * np][];
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < np; j++) {
                    nextRows[i * np + j] = input((int) batchRec[i][2], prefs[j]);
                }
            }
            if (updates++ % 100 == 0) {
                RlTensors.copy(net.parameters(), target.parameters());
            }
            float[][] nextQ = predict(target, nextRows);
            float[][] nextOnline = predict(net, nextRows);
            int rows = batch * np;
            float[] inputs = new float[rows * (in + m)];
            float[] target = new float[rows * na * m];
            float[] mask = new float[rows * na * m];
            for (int i = 0; i < batch; i++) {
                int a = (int) batchRec[i][1];
                boolean term = mdp.terminal()[(int) batchRec[i][2]];
                for (int j = 0; j < np; j++) {
                    double[] w = prefs[j];
                    double best = Double.NEGATIVE_INFINITY;
                    float[] bestVec = null;
                    for (int jj = 0; jj < np; jj++) {
                        float[] q = nextOnline[i * np + jj];
                        for (int b = 0; b < na; b++) {
                            double v = 0;
                            for (int k = 0; k < m; k++) {
                                v += w[k] * q[b * m + k];
                            }
                            if (v > best) {
                                best = v;
                                bestVec = java.util.Arrays.copyOfRange(nextQ[i * np + jj], b * m, (b + 1) * m);
                            }
                        }
                    }
                    int row = i * np + j;
                    System.arraycopy(input((int) batchRec[i][0], w), 0, inputs, row * (in + m), in + m);
                    for (int k = 0; k < m; k++) {
                        double boot = Math.max(low[k], Math.min(high[k], bestVec[k]));
                        double y = batchRec[i][3 + k] + (term ? 0 : gamma * boot);
                        target[row * na * m + a * m + k] = (float) y;
                        mask[row * na * m + a * m + k] = 1;
                    }
                }
            }
            Var pred = net.forward(Var.of(Tensor.of(Shape.of(rows, in + m), inputs)), true);
            Var loss = pred.sub(Var.of(Tensor.of(Shape.of(rows, na * m), target)))
                    .mul(Tensor.of(Shape.of(rows, na * m), mask)).square().sum().mul(1f / rows);
            RlTensors.zeroGrad(net.parameters());
            loss.backward();
            opt.step(net.parameters());
        }

        /**
         * @param w a preference
         * @return the return vector of the greedy policy for it
         */
        public double[] returnFor(double[] w) {
            return rollout(mdp, s -> act(s, w));
        }
    }

    // ------------------------------------------------------------------ helpers

    private static double dot(double[] w, double[] r) {
        double s = 0;
        for (int k = 0; k < w.length; k++) {
            s += w[k] * r[k];
        }
        return s;
    }

    private static double max(double[] v) {
        double m = Double.NEGATIVE_INFINITY;
        for (double x : v) {
            m = Math.max(m, x);
        }
        return m;
    }

    private static int argmax(double[] v, SplittableRandom r) {
        int best = 0;
        int ties = 1;
        for (int a = 1; a < v.length; a++) {
            if (v[a] > v[best] + 1e-12) {
                best = a;
                ties = 1;
            } else if (r != null && Math.abs(v[a] - v[best]) <= 1e-12 && r.nextInt(++ties) == 0) {
                best = a;
            }
        }
        return best;
    }
}
