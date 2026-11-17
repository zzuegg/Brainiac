package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Agent;
import java.util.SplittableRandom;

/**
 * A discrete active-inference agent (Friston et al.; the formulation of Da Costa et al.,
 * "Active inference on discrete state-spaces: a synthesis",
 * <a href="https://arxiv.org/abs/2001.07203">2001.07203</a>). It holds a generative model of a
 * partially observed world: the likelihood {@code A[o][s] = P(o | s)}, the transitions
 * {@code B[a][s'][s] = P(s' | s, a)}, preferred outcomes {@code C[o]} (log-probabilities up to
 * a constant) and the initial state prior {@code D[s]}. It infers its hidden state by Bayesian
 * filtering and picks the action sequence with the least <em>expected free energy</em>
 *
 * <pre>G(pi) = sum_t  E_Q(o_t|pi)[-ln P(o_t | C)]  -  (H[Q(o_t|pi)] - E_Q(s_t|pi) H[P(o_t|s_t)])</pre>
 *
 * The first term (risk) seeks preferred outcomes; the second (epistemic value, the mutual
 * information between states and outcomes) seeks observations that resolve uncertainty. With
 * {@link #epistemic(boolean) epistemic value off} it becomes a purely goal-seeking planner.
 * By default it plans by {@link #sophisticated(boolean) sophisticated inference}, which also
 * anticipates the belief updates to come.
 *
 * <p>Belief state is per episode: {@link #startEpisode()} forgets it, which the core rollouts
 * call before each episode.
 */
public final class ActiveInferenceAgent implements Agent {

    private final double[][] likelihood;
    private final double[][][] transitions;
    private final double[] logPreference;
    private final double[] prior;
    private final int horizon;
    private final SplittableRandom random;
    private boolean epistemic = true;
    private boolean sophisticated = true;
    private double precision = 16;
    private double[] belief;
    private int lastAction = -1;
    private int step;

    private ActiveInferenceAgent(double[][] a, double[][][] b, double[] c, double[] d, int horizon, long seed) {
        this.likelihood = a;
        this.transitions = b;
        this.prior = d.clone();
        this.horizon = horizon;
        this.random = new SplittableRandom(seed);
        double max = Double.NEGATIVE_INFINITY;
        for (double x : c) {
            max = Math.max(max, x);
        }
        double z = 0;
        for (double x : c) {
            z += Math.exp(x - max);
        }
        this.logPreference = new double[c.length];
        for (int o = 0; o < c.length; o++) {
            logPreference[o] = c[o] - max - Math.log(z);
        }
    }

    /**
     * @param a the likelihood {@code A[o][s]}
     * @param b the transitions {@code B[action][s'][s]}
     * @param c preferred outcomes as unnormalized log-probabilities
     * @param d the prior over the initial state
     * @param horizon the planning horizon of an episode, in steps
     * @param seed the seed for sampling actions
     * @return the agent
     */
    public static ActiveInferenceAgent of(double[][] a, double[][][] b, double[] c, double[] d, int horizon,
            long seed) {
        return new ActiveInferenceAgent(a, b, c, d, horizon, seed);
    }

    /**
     * @param on whether expected free energy includes the information gain
     * @return this agent
     */
    public ActiveInferenceAgent epistemic(boolean on) {
        this.epistemic = on;
        return this;
    }

    /**
     * @param on plan by sophisticated inference (Friston et al.,
     *     <a href="https://arxiv.org/abs/2006.04120">2006.04120</a>): a recursive search over
     *     actions and the observations they may bring, each followed by the belief update it
     *     would cause, so the agent values information it will be able to act on. Off, it
     *     scores fixed action sequences under the current belief (classic policies).
     * @return this agent
     */
    public ActiveInferenceAgent sophisticated(boolean on) {
        this.sophisticated = on;
        return this;
    }

    /**
     * @param gamma the policy precision used when sampling ({@code P(pi) ~ exp(-gamma G)})
     * @return this agent
     */
    public ActiveInferenceAgent precision(double gamma) {
        this.precision = gamma;
        return this;
    }

    /** Forgets the belief, before a new episode. */
    @Override
    public void startEpisode() {
        belief = null;
        lastAction = -1;
        step = 0;
    }

    /** @return the current posterior over hidden states */
    public double[] belief() {
        return belief == null ? prior.clone() : belief.clone();
    }

    private double[] predict(double[] q, int a) {
        double[] next = new double[q.length];
        for (int s2 = 0; s2 < q.length; s2++) {
            double v = 0;
            for (int s = 0; s < q.length; s++) {
                v += transitions[a][s2][s] * q[s];
            }
            next[s2] = v;
        }
        return next;
    }

    private void update(int o) {
        double[] p = belief == null ? prior.clone() : predict(belief, lastAction);
        double z = 0;
        for (int s = 0; s < p.length; s++) {
            p[s] *= likelihood[o][s];
            z += p[s];
        }
        for (int s = 0; s < p.length; s++) {
            p[s] = z > 0 ? p[s] / z : 1.0 / p.length;
        }
        belief = p;
    }

    /**
     * The expected free energy of an action sequence from a belief.
     *
     * @param q the belief
     * @param policy the actions
     * @return G
     */
    public double expectedFreeEnergy(double[] q, int[] policy) {
        double g = 0;
        double[] qs = q;
        for (int a : policy) {
            qs = predict(qs, a);
            g += stepFreeEnergy(qs, outcomes(qs));
        }
        return g;
    }

    private double[] outcomes(double[] qs) {
        double[] qo = new double[likelihood.length];
        for (int s = 0; s < qs.length; s++) {
            for (int o = 0; o < likelihood.length; o++) {
                qo[o] += likelihood[o][s] * qs[s];
            }
        }
        return qo;
    }

    private double stepFreeEnergy(double[] qs, double[] qo) {
        double ambiguity = 0;
        for (int s = 0; s < qs.length; s++) {
            double h = 0;
            for (double[] row : likelihood) {
                double p = row[s];
                h -= p > 0 ? p * Math.log(p) : 0;
            }
            ambiguity += qs[s] * h;
        }
        double risk = 0;
        double entropy = 0;
        for (int o = 0; o < qo.length; o++) {
            risk -= qo[o] * logPreference[o];
            entropy -= qo[o] > 0 ? qo[o] * Math.log(qo[o]) : 0;
        }
        return risk - (epistemic ? entropy - ambiguity : 0);
    }

    /**
     * The sophisticated expected free energy of every action: its own step plus, for every
     * outcome it may bring, the best continuation from the belief that outcome would cause.
     *
     * @param q the belief
     * @param depth steps to look ahead
     * @return G per action
     */
    public double[] sophisticatedFreeEnergy(double[] q, int depth) {
        double[] g = new double[transitions.length];
        for (int a = 0; a < g.length; a++) {
            double[] qs = predict(q, a);
            double[] qo = outcomes(qs);
            g[a] = stepFreeEnergy(qs, qo);
            if (depth > 1) {
                for (int o = 0; o < qo.length; o++) {
                    if (qo[o] < 1e-6) {
                        continue;
                    }
                    double[] post = new double[qs.length];
                    for (int s = 0; s < qs.length; s++) {
                        post[s] = qs[s] * likelihood[o][s] / qo[o];
                    }
                    double best = Double.POSITIVE_INFINITY;
                    for (double x : sophisticatedFreeEnergy(post, depth - 1)) {
                        best = Math.min(best, x);
                    }
                    g[a] += qo[o] * best;
                }
            }
        }
        return g;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        update(AgentKit.argmax(observation));
        int depth = Math.max(1, horizon - step);
        int actions = transitions.length;
        if (sophisticated) {
            lastAction = choose(sophisticatedFreeEnergy(belief, depth), explore);
            step++;
            return new float[] {lastAction};
        }
        int count = (int) Math.pow(actions, depth);
        double[] g = new double[count];
        int[] policy = new int[depth];
        for (int p = 0; p < count; p++) {
            int code = p;
            for (int t = 0; t < depth; t++) {
                policy[t] = code % actions;
                code /= actions;
            }
            g[p] = expectedFreeEnergy(belief, policy);
        }
        lastAction = choose(g, explore) % actions;
        step++;
        return new float[] {lastAction};
    }

    private int choose(double[] g, boolean explore) {
        if (explore) {
            float[] logits = new float[g.length];
            for (int p = 0; p < g.length; p++) {
                logits[p] = (float) (-precision * g[p]);
            }
            return AgentKit.sample(logits, random);
        }
        int chosen = 0;
        for (int p = 1; p < g.length; p++) {
            if (g[p] < g[chosen] - 1e-9) {
                chosen = p;
            }
        }
        return chosen;
    }
}
