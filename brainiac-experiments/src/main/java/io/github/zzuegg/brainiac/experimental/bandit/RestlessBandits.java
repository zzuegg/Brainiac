package io.github.zzuegg.brainiac.experimental.bandit;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims.RestlessArm;
import io.github.zzuegg.brainiac.experimental.ops.TwoPhaseSimplex;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Restless bandits: every arm changes state whether pulled or not, and a budget limits how many
 * arms are activated per step. Two-state arms ({@link RestlessArm}: state 1 pays 1) and
 * scheduling policies that activate exactly {@code budget} arms:
 *
 * <ul>
 *   <li>{@link #whittle}: the Whittle index (Whittle, J. Appl. Prob. 1988), per arm and state the
 *       passive subsidy that makes both actions equally good, found by bisection with value
 *       iteration ({@link #whittleIndices}); asymptotically optimal under indexability (Weber &amp;
 *       Weiss 1990);</li>
 *   <li>{@link #lpRelaxation}: the LP relaxation over per-arm occupation measures with the budget
 *       met on average ({@link #lpBound} is an upper bound on any policy's average reward), with
 *       the arms ranked by their LP activation probability;</li>
 *   <li>{@link #whittleQLearning}: Whittle-index Q-learning (Avrachenkov &amp; Borkar, 2022): for
 *       every arm and reference state, Q-learning on the subsidised arm on the fast timescale and
 *       the subsidy moved towards indifference on the slow one, with unknown dynamics;</li>
 *   <li>{@link #neuralIndex}: a NeurWIN-style index (Nakhleh et al., NeurIPS 2021): per arm a small
 *       MLP {@code f(s)} trained by REINFORCE on the simulated subsidised arm, activating with
 *       probability {@code sigmoid((f(s) - lambda) / tau)} for random subsidies {@code lambda};</li>
 *   <li>baselines: {@link #myopic} (largest immediate gain), {@link #roundRobin}, {@link #random}
 *       and {@link #optimal} (dynamic programming on the joint state, 4-6 arms).</li>
 * </ul>
 */
public final class RestlessBandits {

    private RestlessBandits() {
    }

    /** Picks the arms to activate each step and may learn from the outcome. */
    public interface Policy {
        /**
         * @param states every arm's state (0 or 1)
         * @param budget how many arms to activate
         * @return which arms to activate
         */
        boolean[] choose(int[] states, int budget);

        /**
         * Learns from one step (no-op by default).
         *
         * @param states the states before
         * @param active the activation
         * @param next the states after
         */
        default void observe(int[] states, boolean[] active, int[] next) {
        }
    }

    /**
     * Runs a policy and returns the average reward per step (arms in state 1 per step).
     *
     * @param arms the arms
     * @param budget arms activated per step
     * @param policy the policy
     * @param steps steps to run
     * @param seed the seed
     * @return the mean reward per step
     */
    public static double simulate(List<RestlessArm> arms, int budget, Policy policy, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = arms.size();
        int[] s = new int[n];
        for (int i = 0; i < n; i++) {
            s[i] = r.nextInt(2);
        }
        double total = 0;
        for (int t = 0; t < steps; t++) {
            boolean[] act = policy.choose(s.clone(), budget);
            int[] next = new int[n];
            for (int i = 0; i < n; i++) {
                total += s[i];
                double p1 = arms.get(i).transition(act[i] ? 1 : 0)[s[i]][1];
                next[i] = r.nextDouble() < p1 ? 1 : 0;
            }
            policy.observe(s, act, next);
            s = next;
        }
        return total / steps;
    }

    private static boolean[] top(double[] score, int budget, SplittableRandom tieBreak) {
        Integer[] order = new Integer[score.length];
        double[] jitter = new double[score.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
            jitter[i] = tieBreak == null ? 0 : tieBreak.nextDouble() * 1e-9;
        }
        Arrays.sort(order, (x, y) -> Double.compare(score[y] + jitter[y], score[x] + jitter[x]));
        boolean[] act = new boolean[score.length];
        for (int k = 0; k < Math.min(budget, order.length); k++) {
            act[order[k]] = true;
        }
        return act;
    }

    // ------------------------------------------------------------------ Whittle index

    /**
     * Value iteration on one arm with a passive subsidy.
     *
     * @return {@code Q[s][a]}
     */
    private static double[][] subsidisedQ(double[][] p0, double[][] p1, double subsidy, double gamma) {
        double[] v = new double[2];
        double[][] q = new double[2][2];
        for (int it = 0; it < 2000; it++) {
            double change = 0;
            for (int s = 0; s < 2; s++) {
                q[s][0] = s + subsidy + gamma * (p0[s][0] * v[0] + p0[s][1] * v[1]);
                q[s][1] = s + gamma * (p1[s][0] * v[0] + p1[s][1] * v[1]);
            }
            for (int s = 0; s < 2; s++) {
                double nv = Math.max(q[s][0], q[s][1]);
                change = Math.max(change, Math.abs(nv - v[s]));
                v[s] = nv;
            }
            if (change < 1e-10) {
                break;
            }
        }
        return q;
    }

    /**
     * The Whittle index of every state of an arm.
     *
     * @param arm the arm
     * @param gamma the discount
     * @return the subsidy making both actions equally good, per state
     */
    public static double[] whittleIndices(RestlessArm arm, double gamma) {
        double[] index = new double[2];
        for (int s = 0; s < 2; s++) {
            double lo = -2;
            double hi = 2;
            for (int it = 0; it < 50; it++) {
                double mid = (lo + hi) / 2;
                double[][] q = subsidisedQ(arm.passive(), arm.active(), mid, gamma);
                if (q[s][1] > q[s][0]) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            index[s] = (lo + hi) / 2;
        }
        return index;
    }

    /**
     * @param arms the arms
     * @param gamma the discount of the index computation
     * @return the Whittle index policy
     */
    public static Policy whittle(List<RestlessArm> arms, double gamma) {
        double[][] idx = new double[arms.size()][];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = whittleIndices(arms.get(i), gamma);
        }
        SplittableRandom tie = new SplittableRandom(7);
        return (states, budget) -> {
            double[] score = new double[states.length];
            for (int i = 0; i < score.length; i++) {
                score[i] = idx[i][states[i]];
            }
            return top(score, budget, tie);
        };
    }

    // ------------------------------------------------------------------ LP relaxation

    /**
     * Solves the average-reward LP relaxation.
     *
     * @return the occupation measures {@code x[4 arm + 2 s + a]} followed by the objective
     */
    private static double[] lp(List<RestlessArm> arms, int budget) {
        int n = arms.size();
        int vars = 4 * n;
        double[] c = new double[vars];
        double[][] eq = new double[3 * n][vars];
        double[] beq = new double[3 * n];
        for (int i = 0; i < n; i++) {
            int base = 4 * i;
            for (int s = 0; s < 2; s++) {
                for (int a = 0; a < 2; a++) {
                    c[base + 2 * s + a] = s;
                    eq[3 * i][base + 2 * s + a] = 1;
                }
            }
            beq[3 * i] = 1;
            for (int s2 = 0; s2 < 2; s2++) {
                double[] row = eq[3 * i + 1 + s2];
                for (int a = 0; a < 2; a++) {
                    row[base + 2 * s2 + a] += 1;
                }
                for (int s = 0; s < 2; s++) {
                    for (int a = 0; a < 2; a++) {
                        row[base + 2 * s + a] -= arms.get(i).transition(a)[s][s2];
                    }
                }
            }
        }
        double[][] ub = new double[1][vars];
        for (int i = 0; i < n; i++) {
            ub[0][4 * i + 1] = 1;
            ub[0][4 * i + 3] = 1;
        }
        TwoPhaseSimplex.Result res = TwoPhaseSimplex.maximize(c, ub, new double[] {budget}, eq, beq);
        double[] out = Arrays.copyOf(res.x(), vars + 1);
        out[vars] = res.objective();
        return out;
    }

    /**
     * @param arms the arms
     * @param budget the activation budget (met on average)
     * @return the LP upper bound on the average reward per step
     */
    public static double lpBound(List<RestlessArm> arms, int budget) {
        double[] x = lp(arms, budget);
        return x[x.length - 1];
    }

    /**
     * @param arms the arms
     * @param budget the activation budget
     * @return the policy activating the arms with the highest LP activation probability in their state
     */
    public static Policy lpRelaxation(List<RestlessArm> arms, int budget) {
        double[] x = lp(arms, budget);
        double[][] prob = new double[arms.size()][2];
        for (int i = 0; i < prob.length; i++) {
            for (int s = 0; s < 2; s++) {
                double tot = x[4 * i + 2 * s] + x[4 * i + 2 * s + 1];
                prob[i][s] = tot < 1e-12 ? 0.5 : x[4 * i + 2 * s + 1] / tot;
            }
        }
        double[][] tieBreak = new double[arms.size()][];
        for (int i = 0; i < tieBreak.length; i++) {
            tieBreak[i] = whittleIndices(arms.get(i), 0.95);
        }
        return (states, b) -> {
            double[] score = new double[states.length];
            for (int i = 0; i < score.length; i++) {
                score[i] = prob[i][states[i]] + 1e-3 * tieBreak[i][states[i]];
            }
            return top(score, b, null);
        };
    }

    // ------------------------------------------------------------------ baselines

    /**
     * @param arms the arms
     * @return the policy activating the largest gains in next-step adherence
     */
    public static Policy myopic(List<RestlessArm> arms) {
        return (states, budget) -> {
            double[] score = new double[states.length];
            for (int i = 0; i < score.length; i++) {
                RestlessArm a = arms.get(i);
                score[i] = a.active()[states[i]][1] - a.passive()[states[i]][1];
            }
            return top(score, budget, null);
        };
    }

    /** @return the policy cycling through the arms */
    public static Policy roundRobin() {
        int[] next = {0};
        return (states, budget) -> {
            boolean[] act = new boolean[states.length];
            for (int k = 0; k < budget; k++) {
                act[(next[0] + k) % states.length] = true;
            }
            next[0] = (next[0] + budget) % states.length;
            return act;
        };
    }

    /**
     * @param seed the seed
     * @return the policy activating random arms
     */
    public static Policy random(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return (states, budget) -> {
            double[] score = new double[states.length];
            for (int i = 0; i < score.length; i++) {
                score[i] = r.nextDouble();
            }
            return top(score, budget, null);
        };
    }

    /**
     * The optimal policy by value iteration on the joint state (exponential in the arms; 4-6 arms).
     *
     * @param arms the arms
     * @param budget the activation budget
     * @param gamma the discount (close to 1 for the average reward)
     * @return the policy
     */
    public static Policy optimal(List<RestlessArm> arms, int budget, double gamma) {
        int n = arms.size();
        int states = 1 << n;
        int[] subsets = java.util.stream.IntStream.range(0, states).filter(m -> Integer.bitCount(m) == budget)
                .toArray();
        double[] v = new double[states];
        int[] best = new int[states];
        double[][] nextProb = new double[states * subsets.length][];
        for (int s = 0; s < states; s++) {
            for (int k = 0; k < subsets.length; k++) {
                double[] p = new double[states];
                for (int s2 = 0; s2 < states; s2++) {
                    double prob = 1;
                    for (int i = 0; i < n; i++) {
                        int a = (subsets[k] >> i) & 1;
                        double p1 = arms.get(i).transition(a)[(s >> i) & 1][1];
                        prob *= ((s2 >> i) & 1) == 1 ? p1 : 1 - p1;
                    }
                    p[s2] = prob;
                }
                nextProb[s * subsets.length + k] = p;
            }
        }
        for (int it = 0; it < 3000; it++) {
            double change = 0;
            double[] nv = new double[states];
            for (int s = 0; s < states; s++) {
                double bv = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < subsets.length; k++) {
                    double[] p = nextProb[s * subsets.length + k];
                    double e = 0;
                    for (int s2 = 0; s2 < states; s2++) {
                        e += p[s2] * v[s2];
                    }
                    double q = Integer.bitCount(s) + gamma * e;
                    if (q > bv) {
                        bv = q;
                        best[s] = subsets[k];
                    }
                }
                nv[s] = bv;
                change = Math.max(change, Math.abs(bv - v[s]));
            }
            v = nv;
            if (change < 1e-8) {
                break;
            }
        }
        return (st, b) -> {
            int s = 0;
            for (int i = 0; i < n; i++) {
                s |= st[i] << i;
            }
            boolean[] act = new boolean[n];
            for (int i = 0; i < n; i++) {
                act[i] = ((best[s] >> i) & 1) == 1;
            }
            return act;
        };
    }

    // ------------------------------------------------------------------ learned indices

    /**
     * Whittle-index Q-learning with unknown dynamics.
     *
     * @param arms the number of arms
     * @param gamma the discount
     * @param epsilon the exploration rate
     * @param seed the seed
     * @return the learning policy (its indices are the learned subsidies)
     */
    public static Policy whittleQLearning(int arms, double gamma, double epsilon, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][][][] q = new double[arms][2][2][2];
        double[][] lambda = new double[arms][2];
        int[][][] visits = new int[arms][2][2];
        long[] t = {0};
        return new Policy() {
            @Override
            public boolean[] choose(int[] states, int budget) {
                double[] score = new double[states.length];
                for (int i = 0; i < score.length; i++) {
                    score[i] = r.nextDouble() < epsilon ? r.nextDouble() * 4 - 2 : lambda[i][states[i]];
                }
                return top(score, budget, r);
            }

            @Override
            public void observe(int[] states, boolean[] active, int[] next) {
                t[0]++;
                double beta = 1.0 / (1 + t[0] / 50.0);
                for (int i = 0; i < states.length; i++) {
                    int s = states[i];
                    int a = active[i] ? 1 : 0;
                    int n = ++visits[i][s][a];
                    double alpha = 1.0 / Math.pow(1 + n, 0.6);
                    for (int ref = 0; ref < 2; ref++) {
                        double[][] qi = q[i][ref];
                        double target = s + (a == 0 ? lambda[i][ref] : 0)
                                + gamma * Math.max(qi[next[i]][0], qi[next[i]][1]);
                        qi[s][a] += alpha * (target - qi[s][a]);
                        lambda[i][ref] += 0.05 * beta * (qi[ref][1] - qi[ref][0]) * (1 - gamma);
                    }
                }
            }
        };
    }

    /**
     * NeurWIN-style neural indices trained on a simulator of each arm.
     *
     * @param arms the arms (their simulators)
     * @param episodes training episodes per arm
     * @param gamma the discount
     * @param seed the seed
     * @return the index policy
     */
    public static Policy neuralIndex(List<RestlessArm> arms, int episodes, double gamma, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] idx = new double[arms.size()][2];
        for (int i = 0; i < arms.size(); i++) {
            TinyIndexNet net = new TinyIndexNet(8, r);
            RestlessArm arm = arms.get(i);
            for (int ep = 0; ep < episodes; ep++) {
                double lambda = r.nextDouble() * 2 - 1;
                double tau = 0.2;
                int s = r.nextInt(2);
                double ret = 0;
                double disc = 1;
                double[][] grads = net.zeroGrad();
                for (int step = 0; step < 30; step++) {
                    double f = net.forward(s);
                    double p = 1 / (1 + Math.exp(-(f - lambda) / tau));
                    int a = r.nextDouble() < p ? 1 : 0;
                    double dlogp = ((a == 1 ? 1 - p : -p)) / tau;
                    net.accumulate(s, dlogp, grads);
                    ret += disc * (s + (a == 0 ? lambda : 0));
                    disc *= gamma;
                    s = r.nextDouble() < arm.transition(a)[s][1] ? 1 : 0;
                }
                double baseline = net.baseline;
                net.baseline += 0.05 * (ret - baseline);
                net.step(grads, 0.01 * (ret - baseline));
            }
            idx[i][0] = net.forward(0);
            idx[i][1] = net.forward(1);
        }
        SplittableRandom tie = new SplittableRandom(seed + 1);
        return (states, budget) -> {
            double[] score = new double[states.length];
            for (int i = 0; i < score.length; i++) {
                score[i] = idx[i][states[i]];
            }
            return top(score, budget, tie);
        };
    }

    /** A one-hidden-layer tanh MLP on the one-hot state, with hand-written gradients. */
    private static final class TinyIndexNet {
        private final double[][] w1;
        private final double[] b1;
        private final double[] w2;
        private double b2;
        private double baseline;

        TinyIndexNet(int hidden, SplittableRandom r) {
            w1 = new double[hidden][2];
            b1 = new double[hidden];
            w2 = new double[hidden];
            for (int h = 0; h < hidden; h++) {
                w1[h][0] = r.nextGaussian() * 0.5;
                w1[h][1] = r.nextGaussian() * 0.5;
                w2[h] = r.nextGaussian() * 0.5;
            }
        }

        double forward(int s) {
            double out = b2;
            for (int h = 0; h < w2.length; h++) {
                out += w2[h] * Math.tanh(w1[h][s] + b1[h]);
            }
            return out;
        }

        double[][] zeroGrad() {
            return new double[w2.length + 1][4];
        }

        /** Adds {@code coefficient * d f(s) / d params} to the gradient rows (w1[s], b1, w2) per unit, last row b2. */
        void accumulate(int s, double coefficient, double[][] g) {
            for (int h = 0; h < w2.length; h++) {
                double z = Math.tanh(w1[h][s] + b1[h]);
                double dz = w2[h] * (1 - z * z);
                g[h][s] += coefficient * dz;
                g[h][2] += coefficient * dz;
                g[h][3] += coefficient * z;
            }
            g[w2.length][0] += coefficient;
        }

        void step(double[][] g, double lr) {
            for (int h = 0; h < w2.length; h++) {
                w1[h][0] += lr * g[h][0];
                w1[h][1] += lr * g[h][1];
                b1[h] += lr * g[h][2];
                w2[h] += lr * g[h][3];
            }
            b2 += lr * g[w2.length][0];
        }
    }
}
