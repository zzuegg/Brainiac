package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * The stochastic middle of a variational autoencoder (Kingma and Welling,
 * <a href="https://arxiv.org/abs/1312.6114">1312.6114</a>): takes {@code [batch, 2 * latent]}
 * (a mean and a log-variance per latent) and returns a latent sample
 * {@code z = mean + exp(logvar / 2) * eps} with the <em>reparameterization trick</em>, so
 * gradients flow through the sampling. At inference it returns the mean.
 *
 * <p>{@link #kl(Var, float)} is the matching KL divergence to the standard normal prior,
 * with <em>free bits</em> (Kingma et al., <a href="https://arxiv.org/abs/1606.04934">1606.04934</a>):
 * each latent dimension's batch-averaged KL is clamped from below at {@code lambda} nats,
 * so the optimizer gains nothing from pushing a dimension below that. This keeps latents
 * from collapsing onto the prior (posterior collapse).
 *
 * <pre>{@code
 * Model encoder = Model.sequential().input(64).dense(64, Activation.SILU).dense(2 * 4).build();
 * VariationalBottleneck vb = VariationalBottleneck.of(4, 7);
 * Var stats = encoder.forward(x);
 * Var z = vb.forward(stats, true);
 * Var loss = reconstruction.add(VariationalBottleneck.kl(stats, 0.1f));
 * }</pre>
 */
public final class VariationalBottleneck implements Layer {
    private final int latent;
    private final SplittableRandom random;

    private VariationalBottleneck(int latent, long seed) {
        if (latent < 1) {
            throw new IllegalArgumentException("latent must be at least 1");
        }
        this.latent = latent;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates the layer.
     *
     * @param latent the latent size
     * @param seed the seed for the sampling noise
     * @return the layer
     */
    public static VariationalBottleneck of(int latent, long seed) {
        return new VariationalBottleneck(latent, seed);
    }

    /**
     * Describes the layer for model builders: the input's last dimension must be even and
     * becomes half as wide.
     *
     * @param seed the seed for the sampling noise
     * @return the spec
     */
    public static LayerSpec spec(long seed) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), input.dim(-1) / 2);
            }

            @Override
            public Layer create(Shape input, SplittableRandom r) {
                return of(input.dim(-1) / 2, seed);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        check(input);
        Var mean = input.slice(1, 0, latent);
        if (!training) {
            return mean;
        }
        Var logvar = input.slice(1, latent, 2 * latent);
        Tensor eps = Tensor.normal(mean.shape(), 0, 1, random).to(input.value().device());
        return mean.add(logvar.mul(0.5f).exp().mul(eps));
    }

    @Override
    public int outputSize(int inputSize) {
        return inputSize / 2;
    }

    /**
     * KL divergence of {@code N(mean, exp(logvar))} from {@code N(0, 1)}, summed over latent
     * dimensions and averaged over the batch, with free bits.
     *
     * @param stats {@code [batch, 2 * latent]}: means then log-variances
     * @param freeBits the minimum KL per dimension, in nats; 0 for the plain KL
     * @return the scalar
     */
    public static Var kl(Var stats, float freeBits) {
        int latent = stats.shape().dim(1) / 2;
        Var mean = stats.slice(1, 0, latent);
        Var logvar = stats.slice(1, latent, 2 * latent);
        // 0.5 * (mean^2 + var - logvar - 1), per element
        Var perDim = mean.square().add(logvar.exp()).sub(logvar).sub(1f).mul(0.5f).mean(0, false);
        if (freeBits > 0) {
            perDim = perDim.maximum(freeBits);
        }
        return perDim.sum();
    }

    /**
     * The plain KL of each latent dimension, averaged over the batch, for diagnostics such
     * as counting active units.
     *
     * @param stats {@code [batch, 2 * latent]}
     * @return one value per latent dimension
     */
    public static float[] klPerDimension(Tensor stats) {
        int latent = stats.shape().dim(1) / 2;
        Tensor mean = stats.slice(1, 0, latent);
        Tensor logvar = stats.slice(1, latent, 2 * latent);
        return mean.square().add(logvar.exp()).sub(logvar).sub(1f).mul(0.5f).mean(0).toArray();
    }

    private void check(Var input) {
        if (input.shape().rank() != 2 || input.shape().dim(1) != 2 * latent) {
            throw new IllegalArgumentException("VariationalBottleneck(" + latent + ") expects [batch, " + 2 * latent
                    + "], got " + input.shape());
        }
    }
}
