package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The gradient-reversal layer of DANN (Ganin et al., JMLR 2016,
 * <a href="https://arxiv.org/abs/1505.07818">arXiv:1505.07818</a>): the identity on the
 * forward pass, the gradient multiplied by {@code -lambda} on the backward pass, so a feature
 * extractor below it maximises the loss of a domain discriminator above it in the same
 * minimisation. {@link #schedule} is the paper's ramp {@code 2 / (1 + exp(-10 p)) - 1} over
 * training progress {@code p}, without which adversarial adaptation is unstable early on.
 *
 * <p>As a {@link Layer} it holds a mutable {@code lambda} that a training loop updates with
 * the schedule; {@link #apply} is the stateless op.
 */
public final class GradientReversal implements Layer {

    private float lambda;

    /**
     * Creates the layer.
     *
     * @param lambda the initial reversal strength
     */
    public GradientReversal(float lambda) {
        this.lambda = lambda;
    }

    /**
     * @param lambda the reversal strength
     * @return a spec for {@code Model.sequential().add(...)}
     */
    public static LayerSpec spec(float lambda) {
        return LayerSpec.of(new GradientReversal(lambda));
    }

    /**
     * The op: identity forward, {@code -lambda} times the gradient backward.
     *
     * @param x the input
     * @param lambda the reversal strength
     * @return {@code x} with a reversed gradient
     */
    public static Var apply(Var x, float lambda) {
        return Var.custom(v -> v[0].copy(), (in, out, g) -> new Tensor[] {g.mul(-lambda)}, x);
    }

    /**
     * DANN's ramp.
     *
     * @param progress training progress in {@code [0, 1]}
     * @return {@code 2 / (1 + exp(-10 p)) - 1}
     */
    public static float schedule(double progress) {
        return (float) (2 / (1 + Math.exp(-10 * progress)) - 1);
    }

    /** @return the current reversal strength */
    public float lambda() {
        return lambda;
    }

    /** @param lambda the new reversal strength */
    public void lambda(float lambda) {
        this.lambda = lambda;
    }

    @Override
    public Var forward(Var input) {
        return apply(input, lambda);
    }
}
