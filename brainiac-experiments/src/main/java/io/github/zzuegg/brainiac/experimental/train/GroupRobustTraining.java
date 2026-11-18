package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.Function;

/**
 * Training objectives that resist spurious correlations, for data split into environments
 * or groups in which the spurious feature's link to the label differs.
 *
 * <ul>
 *   <li>{@link #irmPenalty} (IRMv1; Arjovsky, Bottou, Gulrajani, Lopez-Paz 2019,
 *       <a href="https://arxiv.org/abs/1907.02893">arXiv:1907.02893</a>): the squared
 *       gradient of an environment's risk with respect to a dummy classifier scale
 *       {@code w = 1} on the logits. It is zero when the same classifier is optimal in every
 *       environment, which a spurious feature whose correlation changes cannot give.</li>
 *   <li>{@link GroupDro} (Sagawa, Koh, Hashimoto, Liang 2020,
 *       <a href="https://arxiv.org/abs/1911.08731">arXiv:1911.08731</a>): minimizes the
 *       worst group's loss through online group weights {@code q_g ~ q_g exp(eta loss_g)},
 *       the loss being {@code sum_g q_g loss_g}.</li>
 *   <li>{@link #vrexPenalty} (V-REx; Krueger et al. 2021,
 *       <a href="https://arxiv.org/abs/2003.00688">arXiv:2003.00688</a>): the variance of
 *       the risks across training environments. Simpler and better conditioned than IRM's
 *       gradient penalty, it pushes the model toward features whose risk is the same in
 *       every environment.</li>
 *   <li>{@link JustTrainTwice} (JTT; Liu et al. 2021,
 *       <a href="https://arxiv.org/abs/2107.09044">arXiv:2107.09044</a>): needs no group
 *       labels. A briefly trained ERM model's training errors (mostly minority-group
 *       examples) are upsampled {@code lambdaUp} times for a second training run.</li>
 * </ul>
 */
public final class GroupRobustTraining {

    private GroupRobustTraining() {
    }

    /**
     * The IRMv1 penalty of one environment.
     *
     * @param logits the model's logits for the environment's batch
     * @param risk the loss as a function of (scaled) logits
     * @return {@code |d risk(w logits) / d w|^2} at {@code w = 1}, differentiable in the model
     */
    public static Var irmPenalty(Var logits, Function<Var, Var> risk) {
        Var w = Var.param(Tensor.full(Shape.of(1), 1f));
        Var g = Var.gradients(risk.apply(logits.mul(w)), List.of(w), true).get(0);
        return g.square().sum();
    }

    /**
     * The V-REx penalty: the (population) variance of the environments' risks.
     *
     * @param risks each environment's mean loss, scalar
     * @return {@code mean_e (R_e - mean R)^2}, differentiable in the model
     */
    public static Var vrexPenalty(List<Var> risks) {
        Var mean = risks.get(0);
        for (int e = 1; e < risks.size(); e++) {
            mean = mean.add(risks.get(e));
        }
        mean = mean.mul(1f / risks.size());
        Var out = null;
        for (Var r : risks) {
            Var dev = r.sub(mean).square();
            out = out == null ? dev : out.add(dev);
        }
        return out.mul(1f / risks.size());
    }

    /**
     * Just Train Twice: builds the second run's sampling distribution from the first run's
     * mistakes. The caller trains an identification model (plain ERM, few steps), marks the
     * training examples it gets wrong with {@link #errorSet}, and trains a fresh model on
     * minibatches drawn by {@link #sample}, in which every error-set example appears
     * {@code lambdaUp} times.
     */
    public static final class JustTrainTwice {

        private final int[] pool;

        /**
         * Builds the upsampled index pool.
         *
         * @param errors which training examples the identification model misclassified
         * @param lambdaUp how many times to repeat each of them, such as 20
         */
        public JustTrainTwice(boolean[] errors, int lambdaUp) {
            int n = 0;
            for (boolean e : errors) {
                n += e ? lambdaUp : 1;
            }
            pool = new int[n];
            int k = 0;
            for (int i = 0; i < errors.length; i++) {
                for (int c = 0; c < (errors[i] ? lambdaUp : 1); c++) {
                    pool[k++] = i;
                }
            }
        }

        /**
         * Marks misclassified examples.
         *
         * @param predicted predicted classes
         * @param labels true classes
         * @return {@code predicted[i] != labels[i]}
         */
        public static boolean[] errorSet(int[] predicted, int[] labels) {
            boolean[] out = new boolean[labels.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = predicted[i] != labels[i];
            }
            return out;
        }

        /**
         * Draws a minibatch of example indices from the upsampled pool.
         *
         * @param size the batch size
         * @param random the random source
         * @return indices into the training set
         */
        public int[] sample(int size, java.util.random.RandomGenerator random) {
            int[] out = new int[size];
            for (int i = 0; i < size; i++) {
                out[i] = pool[random.nextInt(pool.length)];
            }
            return out;
        }

        /** @return the size of the upsampled pool */
        public int poolSize() {
            return pool.length;
        }
    }

    /** Group distributionally robust optimization's online group weights. */
    public static final class GroupDro {

        private final double[] q;
        private final double eta;

        /**
         * Creates the weights, uniform at first.
         *
         * @param groups how many groups
         * @param eta the step size of the weight update, such as 0.01
         */
        public GroupDro(int groups, double eta) {
            this.q = new double[groups];
            java.util.Arrays.fill(q, 1.0 / groups);
            this.eta = eta;
        }

        /**
         * Updates the weights with this batch's group losses and returns the weighted loss.
         * A group absent from the batch passes {@code null} and keeps its weight.
         *
         * @param groupLosses each group's mean loss in the batch, or null
         * @return {@code sum_g q_g loss_g} over the present groups
         */
        public Var loss(List<Var> groupLosses) {
            double sum = 0;
            for (int g = 0; g < q.length; g++) {
                if (groupLosses.get(g) != null) {
                    q[g] *= Math.exp(eta * groupLosses.get(g).value().item());
                }
                sum += q[g];
            }
            Var out = null;
            for (int g = 0; g < q.length; g++) {
                q[g] /= sum;
                Var l = groupLosses.get(g);
                if (l != null) {
                    Var term = l.mul((float) q[g]);
                    out = out == null ? term : out.add(term);
                }
            }
            return out;
        }

        /** @return the current group weights; a copy */
        public double[] weights() {
            return q.clone();
        }
    }
}
