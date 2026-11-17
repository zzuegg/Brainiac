package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A dense layer with a mean-field Gaussian posterior over its weights, trained by
 * <b>Bayes by backprop</b>: every weight has a mean {@code mu} and a scale
 * {@code sigma = softplus(rho)}; each forward pass samples the weights by the
 * reparameterization {@code w = mu + sigma * eps}, so gradients reach {@code mu} and
 * {@code rho}. The KL divergence from the Gaussian prior {@code N(0, prior^2)} is exposed as
 * the layer's {@link #auxiliaryLoss()} (scaled by {@code 1 / datasetSize}, the per-example
 * share of the ELBO), so a core trainer that adds {@code Model.auxiliaryLoss()} trains the
 * full ELBO. At inference each forward pass is one posterior sample; average several.
 *
 * <p>Uses the local reparameterization trick (Kingma et al. 2015): instead of sampling a
 * weight matrix it samples the pre-activations, {@code N(x mu, x^2 sigma^2)}, which is
 * equivalent in distribution and has lower-variance gradients.
 *
 * <p>Paper: Blundell, Cornebise, Kavukcuoglu, Wierstra, "Weight Uncertainty in Neural
 * Networks" (2015), <a href="https://arxiv.org/abs/1505.05424">arXiv:1505.05424</a>.
 */
public final class BayesianDense implements Layer {

    private final int inputs;
    private final int units;
    private final Activation activation;
    private final Var mu;
    private final Var rho;
    private final Var biasMu;
    private final Var biasRho;
    private final float prior;
    private final float klScale;
    private final SplittableRandom noise;
    private Var kl;

    /**
     * Creates the layer.
     *
     * @param inputs input features
     * @param units output features
     * @param activation the activation
     * @param prior the prior standard deviation of every weight
     * @param datasetSize the training-set size, which scales the KL term per example
     * @param random the initializer, also seeds the sampling
     */
    public BayesianDense(int inputs, int units, Activation activation, float prior, int datasetSize,
            RandomGenerator random) {
        this.inputs = inputs;
        this.units = units;
        this.activation = activation;
        this.mu = Var.param(Tensor.normal(Shape.of(inputs, units), 0f, (float) Math.sqrt(2.0 / inputs), random));
        this.rho = Var.param(Tensor.full(Shape.of(inputs, units), -5f));
        this.biasMu = Var.param(Tensor.zeros(Shape.of(units)));
        this.biasRho = Var.param(Tensor.full(Shape.of(units), -5f));
        this.prior = prior;
        this.klScale = 1f / datasetSize;
        this.noise = new SplittableRandom(random.nextLong());
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param units output features
     * @param activation the activation
     * @param prior the prior standard deviation
     * @param datasetSize the training-set size
     * @return the spec
     */
    public static LayerSpec spec(int units, Activation activation, float prior, int datasetSize) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(units);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new BayesianDense(input.dim(-1), units, activation, prior, datasetSize, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    /**
     * Samples the weights in both modes (the posterior is the model); while training it also
     * records the KL term.
     *
     * @param input {@code [batch, inputs]}
     * @param training whether this is a training pass
     * @return {@code [batch, units]}
     */
    @Override
    public Var forward(Var input, boolean training) {
        Var sigma = rho.softplus();
        Var bSigma = biasRho.softplus();
        Var mean = input.matmul(mu).add(biasMu);
        Var var = input.square().matmul(sigma.square()).add(bSigma.square());
        Tensor eps = Tensor.normal(mean.shape(), 0f, 1f, noise);
        Var pre = mean.add(var.add(1e-12f).sqrt().mul(eps));
        if (training) {
            kl = klTo(mu, sigma).add(klTo(biasMu, bSigma)).mul(klScale);
        }
        return activation.forward(pre);
    }

    /** {@code KL(N(m, s^2) || N(0, prior^2))} summed over the weights. */
    private Var klTo(Var m, Var s) {
        float p2 = prior * prior;
        return s.square().add(m.square()).div(2 * p2).sub(s.log()).add((float) Math.log(prior) - 0.5f).sum();
    }

    /** @return the KL term of the last training pass, already divided by the dataset size */
    @Override
    public Var auxiliaryLoss() {
        return kl;
    }

    @Override
    public List<Var> parameters() {
        return List.of(mu, rho, biasMu, biasRho);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    /** @return the mean posterior weight scale, a measure of how uncertain the layer is */
    public float meanSigma() {
        return rho.value().softplus().mean().item();
    }

    @Override
    public String toString() {
        return "BayesianDense(" + inputs + " -> " + units + ", " + activation + ")";
    }
}
