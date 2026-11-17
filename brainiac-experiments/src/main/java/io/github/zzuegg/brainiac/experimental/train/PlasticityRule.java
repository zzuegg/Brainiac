package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A local synaptic plasticity rule: the weight change of a {@code [in, out]} weight matrix
 * from the activity before ({@code pre}, {@code [batch, in]}) and after ({@code post},
 * {@code [batch, out]}) the synapse, averaged over the batch. Rules are reusable for any
 * dense weight, for example a {@code Dense} layer's or a {@code Brain} projection's, and
 * can be evolved as parameters of a learning system.
 *
 * <ul>
 *   <li>{@link #hebb}: "fire together, wire together", {@code dW = rate pre^T post}
 *       (Hebb 1949); unstable on its own.</li>
 *   <li>{@link #oja}: Hebb with a decay that keeps weights normalised; one unit converges
 *       to the first principal component (Oja 1982). With several units this is Oja's
 *       subspace rule.</li>
 *   <li>{@link #sanger}: the generalised Hebbian algorithm, whose units converge to the
 *       leading principal components in order (Sanger 1989).</li>
 *   <li>{@link #bcm}: Bienenstock-Cooper-Munro, {@code dW = rate pre^T (post (post - theta))}
 *       with a sliding threshold {@code theta} that tracks the mean squared activity, so
 *       units become selective (BCM 1982).</li>
 *   <li>{@link #covariance}: Hebb on deviations from running means, which can also weaken
 *       synapses (Sejnowski 1977).</li>
 * </ul>
 */
@FunctionalInterface
public interface PlasticityRule {

    /**
     * The weight change for one batch.
     *
     * @param pre presynaptic activity, {@code [batch, in]}
     * @param post postsynaptic activity, {@code [batch, out]}
     * @param weight the current weights, {@code [in, out]}
     * @return the change, {@code [in, out]}
     */
    Tensor update(Tensor pre, Tensor post, Tensor weight);

    /**
     * Applies the change to a weight in place.
     *
     * @param weight the weight
     * @param pre presynaptic activity
     * @param post postsynaptic activity
     */
    default void apply(Var weight, Tensor pre, Tensor post) {
        weight.assign(weight.value().add(update(pre, post, weight.value())));
    }

    /**
     * @param rate the learning rate
     * @return plain Hebbian learning
     */
    static PlasticityRule hebb(float rate) {
        return (pre, post, w) -> pre.transpose().matmul(post).mul(rate / pre.shape().dim(0));
    }

    /**
     * @param rate the learning rate
     * @return Oja's rule, {@code dW = rate (pre^T post - W post^T post)}
     */
    static PlasticityRule oja(float rate) {
        return (pre, post, w) -> pre.transpose().matmul(post).sub(w.matmul(post.transpose().matmul(post)))
                .mul(rate / pre.shape().dim(0));
    }

    /**
     * @param rate the learning rate
     * @return Sanger's rule, {@code dW = rate (pre^T post - W upper(post^T post))}
     */
    static PlasticityRule sanger(float rate) {
        return (pre, post, w) -> {
            int out = post.shape().dim(1);
            float[] m = new float[out * out];
            for (int k = 0; k < out; k++) {
                for (int j = k; j < out; j++) {
                    m[k * out + j] = 1;
                }
            }
            Tensor upper = post.transpose().matmul(post).mul(Tensor.of(Shape.of(out, out), m));
            return pre.transpose().matmul(post).sub(w.matmul(upper)).mul(rate / pre.shape().dim(0));
        };
    }

    /**
     * The BCM rule with a per-unit sliding threshold; the returned rule keeps that state.
     *
     * @param rate the learning rate
     * @param thresholdRate how fast the threshold tracks {@code post^2}, for example 0.1
     * @return the rule
     */
    static PlasticityRule bcm(float rate, float thresholdRate) {
        return new PlasticityRule() {
            private Tensor theta;

            @Override
            public Tensor update(Tensor pre, Tensor post, Tensor w) {
                Tensor meanSq = post.square().mean(0, true);
                theta = theta == null ? meanSq : theta.add(meanSq.sub(theta).mul(thresholdRate));
                Tensor.evaluate(java.util.List.of(theta));
                Tensor phi = post.mul(post.sub(theta));
                return pre.transpose().matmul(phi).mul(rate / pre.shape().dim(0));
            }
        };
    }

    /**
     * The covariance rule with running means of both activities; the rule keeps them.
     *
     * @param rate the learning rate
     * @param meanRate how fast the means track the activity, for example 0.1
     * @return the rule
     */
    static PlasticityRule covariance(float rate, float meanRate) {
        return new PlasticityRule() {
            private Tensor preMean;
            private Tensor postMean;

            @Override
            public Tensor update(Tensor pre, Tensor post, Tensor w) {
                Tensor a = pre.mean(0, true);
                Tensor b = post.mean(0, true);
                preMean = preMean == null ? a : preMean.add(a.sub(preMean).mul(meanRate));
                postMean = postMean == null ? b : postMean.add(b.sub(postMean).mul(meanRate));
                Tensor.evaluate(java.util.List.of(preMean, postMean));
                return pre.sub(preMean).transpose().matmul(post.sub(postMean)).mul(rate / pre.shape().dim(0));
            }
        };
    }
}
