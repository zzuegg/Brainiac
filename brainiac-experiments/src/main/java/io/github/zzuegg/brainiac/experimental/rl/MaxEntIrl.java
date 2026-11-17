package io.github.zzuegg.brainiac.experimental.rl;

import java.util.List;
import java.util.Objects;

/**
 * Maximum-entropy inverse reinforcement learning (Ziebart et al., 2008,
 * <a href="https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf">AAAI 2008</a>): recovers a reward
 * function that explains expert demonstrations in a {@link TabularMdp} with known dynamics.
 *
 * <p>The reward is linear in state features, {@code r(s) = theta . phi(s)}. Experts are
 * modelled as soft-optimal: trajectories are exponentially more likely the more reward
 * they collect. Maximizing the demonstrations' likelihood gives the gradient
 * {@code E_expert[phi] - E_model[phi]}: the expert's feature counts minus the expected
 * feature counts of the soft-optimal policy under the current reward, computed exactly by
 * soft value iteration (backward) and a state-visitation pass (forward). Inverse RL
 * matters because a reward generalizes where a cloned policy does not: to new dynamics,
 * and as a compact explanation of behaviour.
 */
public final class MaxEntIrl {

    private final TabularMdp mdp;
    private final double[][] features;
    private double gamma = 0.99;
    private double learningRate = 0.1;
    private double[] theta;

    private MaxEntIrl(TabularMdp mdp, double[][] features) {
        this.mdp = mdp;
        this.features = features;
        this.theta = new double[features[0].length];
    }

    /**
     * IRL with one indicator feature per state (a free reward per state).
     *
     * @param mdp the dynamics (its own reward is ignored)
     * @return the learner
     */
    public static MaxEntIrl of(TabularMdp mdp) {
        double[][] f = new double[mdp.states()][mdp.states()];
        for (int s = 0; s < f.length; s++) {
            f[s][s] = 1;
        }
        return new MaxEntIrl(mdp, f);
    }

    /**
     * @param mdp the dynamics
     * @param features {@code phi[s]}, one feature vector per state
     * @return the learner
     */
    public static MaxEntIrl of(TabularMdp mdp, double[][] features) {
        if (features.length != mdp.states()) {
            throw new IllegalArgumentException("need one feature vector per state");
        }
        return new MaxEntIrl(Objects.requireNonNull(mdp), features);
    }

    /** @param discount the discount of the soft-optimal model @return this learner */
    public MaxEntIrl discount(double discount) {
        this.gamma = discount;
        return this;
    }

    /** @param lr the gradient-ascent step @return this learner */
    public MaxEntIrl learningRate(double lr) {
        this.learningRate = lr;
        return this;
    }

    /** @return the learned reward of every state */
    public double[] reward() {
        double[] r = new double[mdp.states()];
        for (int s = 0; s < r.length; s++) {
            for (int f = 0; f < theta.length; f++) {
                r[s] += theta[f] * features[s][f];
            }
        }
        return r;
    }

    /** @return the learned feature weights */
    public double[] weights() {
        return theta.clone();
    }

    /**
     * Expert feature counts: average over trajectories of the features of every state
     * visited, start included.
     */
    private double[] expertCounts(OfflineDataset demos) {
        double[] mu = new double[theta.length];
        List<List<OfflineDataset.Step>> trajs = demos.trajectories();
        for (List<OfflineDataset.Step> t : trajs) {
            add(mu, RlTensors.argmax(t.getFirst().observation()));
            for (OfflineDataset.Step s : t) {
                add(mu, RlTensors.argmax(s.nextObservation()));
            }
        }
        for (int f = 0; f < mu.length; f++) {
            mu[f] /= trajs.size();
        }
        return mu;
    }

    private void add(double[] mu, int state) {
        for (int f = 0; f < mu.length; f++) {
            mu[f] += features[state][f];
        }
    }

    /**
     * @return the soft-optimal policy {@code pi[s][a]} under the current reward
     */
    public double[][] policy() {
        double[][] q = mdp.withReward(reward()).softQ(gamma, mdp.maxSteps(), 1);
        double[][] pi = new double[q.length][];
        for (int s = 0; s < q.length; s++) {
            double max = Double.NEGATIVE_INFINITY;
            for (double x : q[s]) {
                max = Math.max(max, x);
            }
            pi[s] = new double[q[s].length];
            double z = 0;
            for (int a = 0; a < q[s].length; a++) {
                pi[s][a] = Math.exp(q[s][a] - max);
                z += pi[s][a];
            }
            for (int a = 0; a < q[s].length; a++) {
                pi[s][a] /= z;
            }
        }
        return pi;
    }

    /** Expected feature counts of the soft-optimal policy over the horizon. */
    private double[] modelCounts(double[][] pi) {
        int n = mdp.states();
        double[] d = mdp.start();
        double[] mu = new double[theta.length];
        for (int s = 0; s < n; s++) {
            for (int f = 0; f < mu.length; f++) {
                mu[f] += d[s] * features[s][f];
            }
        }
        for (int t = 0; t < mdp.maxSteps(); t++) {
            double[] next = new double[n];
            for (int s = 0; s < n; s++) {
                if (d[s] == 0 || mdp.terminal(s)) {
                    continue;
                }
                for (int a = 0; a < pi[s].length; a++) {
                    double[] p = mdp.transition(s, a);
                    for (int u = 0; u < n; u++) {
                        next[u] += d[s] * pi[s][a] * p[u];
                    }
                }
            }
            for (int s = 0; s < n; s++) {
                for (int f = 0; f < mu.length; f++) {
                    mu[f] += next[s] * features[s][f];
                }
            }
            d = next;
        }
        return mu;
    }

    /**
     * Fits the reward to demonstrations.
     *
     * @param demos expert trajectories on the MDP (one-hot state observations)
     * @param iterations gradient steps
     * @return the final feature-count gap (L1), 0 when expert and model match
     */
    public double fit(OfflineDataset demos, int iterations) {
        double[] expert = expertCounts(demos);
        double gap = Double.NaN;
        for (int it = 0; it < iterations; it++) {
            double[] model = modelCounts(policy());
            gap = 0;
            for (int f = 0; f < theta.length; f++) {
                double g = expert[f] - model[f];
                theta[f] += learningRate * g;
                gap += Math.abs(g);
            }
        }
        return gap;
    }
}
