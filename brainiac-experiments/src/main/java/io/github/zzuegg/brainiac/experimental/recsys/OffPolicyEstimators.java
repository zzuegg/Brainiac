package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Off-policy evaluation and learning from logged bandit feedback: estimate the reward a new
 * policy {@code pi} would earn from logs {@code (x_i, a_i, r_i, mu(a_i | x_i))} collected by an old
 * logging policy {@code mu}, and learn a better policy from them, without deploying anything.
 * <ul>
 *   <li>{@link #directMethod}: average a reward model's prediction under {@code pi} (biased when
 *       the model is wrong);</li>
 *   <li>{@link #ips}: inverse propensity scoring, {@code mean(w_i r_i)} with
 *       {@code w_i = pi(a_i | x_i) / mu(a_i | x_i)}, optionally clipped (unbiased, high
 *       variance);</li>
 *   <li>{@link #snips}: self-normalised IPS, {@code sum(w r) / sum(w)} (Swaminathan and Joachims,
 *       NeurIPS 2015);</li>
 *   <li>{@link #doublyRobust}: the reward model as a control variate,
 *       {@code DM + mean(w_i (r_i - q(x_i, a_i)))} (Dudík et al., 2011,
 *       <a href="https://arxiv.org/abs/1103.4601">arXiv:1103.4601</a>), unbiased if either the
 *       propensities or the model are right;</li>
 *   <li>{@link #switchDr}: the Switch estimator (Wang et al., 2017,
 *       <a href="https://arxiv.org/abs/1612.01205">arXiv:1612.01205</a>): importance weighting where
 *       {@code pi / mu <= tau}, the reward model for actions with larger weights;</li>
 *   <li>{@link #learnPolicy}: counterfactual risk minimisation (POEM-style): a linear softmax
 *       policy maximising the clipped IPS estimate minus a variance penalty.</li>
 * </ul>
 * The existing {@code CausalEstimators.ipw/aipw} estimate average treatment effects of a binary
 * treatment; these estimate the value of a policy over many actions.
 */
public final class OffPolicyEstimators {

    private OffPolicyEstimators() {
    }

    /**
     * A ridge-regression reward model, one linear model with bias per action.
     */
    public static final class RewardModel {
        private final double[][] weights;

        private RewardModel(double[][] weights) {
            this.weights = weights;
        }

        /**
         * Fits one ridge regression per action on that action's logged rounds.
         *
         * @param log the logs
         * @param actions the number of actions
         * @param lambda the ridge penalty
         * @return the model
         */
        public static RewardModel fit(SyntheticRecsys.BanditLog log, int actions, double lambda) {
            int d = log.contexts()[0].length + 1;
            double[][] w = new double[actions][];
            for (int a = 0; a < actions; a++) {
                double[][] gram = MlMatrices.identity(d);
                for (int i = 0; i < d; i++) {
                    gram[i][i] = lambda;
                }
                double[] rhs = new double[d];
                for (int i = 0; i < log.rounds(); i++) {
                    if (log.actions()[i] != a) {
                        continue;
                    }
                    double[] x = withBias(log.contexts()[i]);
                    for (int p = 0; p < d; p++) {
                        rhs[p] += x[p] * log.rewards()[i];
                        for (int q = 0; q < d; q++) {
                            gram[p][q] += x[p] * x[q];
                        }
                    }
                }
                w[a] = MlMatrices.choleskySolve(MlMatrices.cholesky(gram), rhs);
            }
            return new RewardModel(w);
        }

        /**
         * @param x a context
         * @param a an action
         * @return the predicted reward
         */
        public double predict(double[] x, int a) {
            return MlMatrices.dot(weights[a], withBias(x));
        }

        /**
         * @param log logs
         * @return {@code [rounds][actions]} predictions at the logged contexts
         */
        public double[][] predict(SyntheticRecsys.BanditLog log) {
            double[][] q = new double[log.rounds()][weights.length];
            for (int i = 0; i < q.length; i++) {
                for (int a = 0; a < weights.length; a++) {
                    q[i][a] = predict(log.contexts()[i], a);
                }
            }
            return q;
        }
    }

    private static double[] withBias(double[] x) {
        double[] out = java.util.Arrays.copyOf(x, x.length + 1);
        out[x.length] = 1;
        return out;
    }

    /**
     * @param log logs
     * @return the average logged reward, the logging policy's value (biased for any other policy)
     */
    public static double naive(SyntheticRecsys.BanditLog log) {
        double s = 0;
        for (double r : log.rewards()) {
            s += r;
        }
        return s / log.rounds();
    }

    /**
     * @param log logs
     * @param pi {@code [rounds][actions]} the target policy at the logged contexts
     * @param q {@code [rounds][actions]} a reward model's predictions at the logged contexts
     * @return the direct-method estimate
     */
    public static double directMethod(SyntheticRecsys.BanditLog log, double[][] pi, double[][] q) {
        double s = 0;
        for (int i = 0; i < log.rounds(); i++) {
            for (int a = 0; a < pi[i].length; a++) {
                s += pi[i][a] * q[i][a];
            }
        }
        return s / log.rounds();
    }

    private static double weight(SyntheticRecsys.BanditLog log, double[][] pi, int i) {
        return pi[i][log.actions()[i]] / log.propensities()[i];
    }

    /**
     * @param log logs
     * @param pi the target policy at the logged contexts
     * @param clip the largest weight, or {@code Double.POSITIVE_INFINITY} for none
     * @return the (clipped) IPS estimate
     */
    public static double ips(SyntheticRecsys.BanditLog log, double[][] pi, double clip) {
        double s = 0;
        for (int i = 0; i < log.rounds(); i++) {
            s += Math.min(clip, weight(log, pi, i)) * log.rewards()[i];
        }
        return s / log.rounds();
    }

    /**
     * @param log logs
     * @param pi the target policy at the logged contexts
     * @return the self-normalised IPS estimate
     */
    public static double snips(SyntheticRecsys.BanditLog log, double[][] pi) {
        double s = 0;
        double z = 0;
        for (int i = 0; i < log.rounds(); i++) {
            double w = weight(log, pi, i);
            s += w * log.rewards()[i];
            z += w;
        }
        return z == 0 ? 0 : s / z;
    }

    /**
     * @param log logs
     * @param pi the target policy at the logged contexts
     * @param q a reward model's predictions at the logged contexts
     * @return the doubly robust estimate
     */
    public static double doublyRobust(SyntheticRecsys.BanditLog log, double[][] pi, double[][] q) {
        double s = directMethod(log, pi, q) * log.rounds();
        for (int i = 0; i < log.rounds(); i++) {
            int a = log.actions()[i];
            s += weight(log, pi, i) * (log.rewards()[i] - q[i][a]);
        }
        return s / log.rounds();
    }

    /**
     * The Switch estimator: importance weighting for actions whose weight
     * {@code pi(a | x) / mu(a | x)} is at most {@code tau}, the reward model for the others. Needs
     * the logging policy's full distribution ({@code log.logging()}).
     *
     * @param log logs
     * @param pi the target policy at the logged contexts
     * @param q a reward model's predictions at the logged contexts
     * @param tau the weight threshold
     * @return the estimate
     */
    public static double switchDr(SyntheticRecsys.BanditLog log, double[][] pi, double[][] q, double tau) {
        double s = 0;
        for (int i = 0; i < log.rounds(); i++) {
            int a = log.actions()[i];
            double w = weight(log, pi, i);
            if (w <= tau) {
                s += w * log.rewards()[i];
            }
            for (int b = 0; b < pi[i].length; b++) {
                if (pi[i][b] / log.logging()[i][b] > tau) {
                    s += pi[i][b] * q[i][b];
                }
            }
        }
        return s / log.rounds();
    }

    /**
     * Learns a linear softmax policy {@code pi(a | x) = softmax(x W)} by counterfactual risk
     * minimisation: maximise {@code mean(min(clip, w_i) r_i) - lambda sqrt(var / n)} with Adam.
     *
     * @param log logs
     * @param actions the number of actions
     * @param clip the weight clip {@code M}
     * @param lambda the variance penalty
     * @param steps full-batch Adam steps
     * @param seed the initialization seed
     * @return {@code [dim + 1][actions]} weights (the last row is the bias), for {@link #policy}
     */
    public static double[][] learnPolicy(SyntheticRecsys.BanditLog log, int actions, double clip, double lambda,
            int steps, long seed) {
        int n = log.rounds();
        int d = log.contexts()[0].length + 1;
        float[] x = new float[n * d];
        float[] taken = new float[n * actions];
        float[] invProp = new float[n];
        float[] reward = new float[n];
        for (int i = 0; i < n; i++) {
            double[] xb = withBias(log.contexts()[i]);
            for (int j = 0; j < d; j++) {
                x[i * d + j] = (float) xb[j];
            }
            taken[i * actions + log.actions()[i]] = 1;
            invProp[i] = (float) (1 / log.propensities()[i]);
            reward[i] = (float) log.rewards()[i];
        }
        Tensor xs = Tensor.of(Shape.of(n, d), x);
        Tensor mask = Tensor.of(Shape.of(n, actions), taken);
        Tensor inv = Tensor.of(Shape.of(n, 1), invProp);
        Tensor r = Tensor.of(Shape.of(n, 1), reward);
        Var w = Var.param(Tensor.normal(Shape.of(d, actions), 0, 0.01f, new SplittableRandom(seed)));
        Optimizer opt = Optimizer.adam(0.05f);
        for (int s = 0; s < steps; s++) {
            Var p = Activation.SOFTMAX.forward(Var.of(xs).matmul(w)).mul(mask).sum(1, true);
            Var term = p.mul(inv).minimum((float) clip).mul(r);
            Var mean = term.mean();
            Var var = term.sub(mean).square().mean();
            mean.neg().add(var.add(1e-8f).sqrt().mul((float) (lambda / Math.sqrt(n)))).backward();
            opt.step(List.of(w));
        }
        float[] wv = w.value().toArray();
        double[][] out = new double[d][actions];
        for (int i = 0; i < d; i++) {
            for (int a = 0; a < actions; a++) {
                out[i][a] = wv[i * actions + a];
            }
        }
        return out;
    }

    /**
     * @param weights weights from {@link #learnPolicy}
     * @param x a context
     * @return the policy's action probabilities
     */
    public static double[] policy(double[][] weights, double[] x) {
        int actions = weights[0].length;
        double[] xb = withBias(x);
        double[] z = new double[actions];
        double max = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < actions; a++) {
            for (int j = 0; j < xb.length; j++) {
                z[a] += xb[j] * weights[j][a];
            }
            max = Math.max(max, z[a]);
        }
        double sum = 0;
        for (int a = 0; a < actions; a++) {
            z[a] = Math.exp(z[a] - max);
            sum += z[a];
        }
        for (int a = 0; a < actions; a++) {
            z[a] /= sum;
        }
        return z;
    }
}
