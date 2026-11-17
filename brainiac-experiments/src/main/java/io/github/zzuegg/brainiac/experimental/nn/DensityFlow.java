package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Flows;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A normalizing flow over vectors {@code [batch, dimension]} with a standard normal base: an
 * invertible map to latents with a tractable log-determinant, so {@link #logProb} is the
 * exact log-density and {@link #sample} is the inverse pass from Gaussian noise. Core
 * {@link Flows} chains only affine {@link Flows.Coupling}s; this interface is the general
 * step that Glow's ActNorm and 1x1 mixing, spline couplings and autoregressive flows need,
 * and it returns the core {@link Flows.Result}, so {@link Flows#logLikelihood} applies.
 */
public interface DensityFlow {

    /**
     * Maps data to latents.
     *
     * @param x data {@code [batch, dimension]}
     * @return the latents and {@code log |det dz/dx|} per row
     */
    Flows.Result forward(Var x);

    /**
     * Maps latents back to data; the exact inverse of {@link #forward}, not differentiated.
     *
     * @param z latents {@code [batch, dimension]}
     * @return data {@code [batch, dimension]}
     */
    Tensor inverse(Tensor z);

    /** @return the data dimension */
    int dimension();

    /** @return every trainable parameter */
    List<Var> parameters();

    /**
     * The exact log-density of each row.
     *
     * @param x data {@code [batch, dimension]}
     * @return {@code [batch]}
     */
    default Var logProb(Var x) {
        return Flows.logLikelihood(forward(x));
    }

    /**
     * Draws samples.
     *
     * @param n how many
     * @param random the noise source
     * @return {@code [n, dimension]}
     */
    default Tensor sample(int n, RandomGenerator random) {
        return inverse(Tensor.normal(Shape.of(n, dimension()), 0, 1, random));
    }
}
