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
