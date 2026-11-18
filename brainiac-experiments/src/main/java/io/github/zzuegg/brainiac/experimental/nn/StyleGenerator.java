package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/**
 * A StyleGAN-lite generator (Karras et al., <a href="https://arxiv.org/abs/1812.04948">arXiv:1812.04948</a>
 * and StyleGAN2, <a href="https://arxiv.org/abs/1912.04958">arXiv:1912.04958</a>) for flat
 * outputs such as small images:
 * <ul>
 *   <li>a <b>mapping network</b> {@code z -> w} (an MLP on the normalised latent);</li>
 *   <li>a learned constant input followed by <b>modulated, demodulated</b> dense layers: each
 *       layer's weight is scaled per input by the style {@code s = A(w)} and every output is
 *       renormalised to unit expected scale, {@code y_j = sum_i s_i W_ij x_i / sqrt(sum_i
 *       (s_i W_ij)^2)} (StyleGAN2's replacement of AdaIN, here for dense layers);</li>
 *   <li><b>noise injection</b>: per-unit Gaussian noise with learned strength after every layer;</li>
 *   <li><b>style mixing</b>: with a given probability a second latent drives the layers after
 *       a random crossover;</li>
 *   <li>the <b>path-length regulariser</b> {@link #pathLengthPenalty}: {@code (|J_w^T y| - a)^2}
 *       for image-space noise {@code y}, with {@code a} a running mean, which makes the map
 *       from {@code w} smoother (lower perceptual path length).</li>
 * </ul>
 * As a {@code Function<Var, Var>} it maps {@code [n, latent]} noise (or {@code [n, 2 latent]},
 * the second half for mixing) to tanh outputs, for core {@code Adversarial}.
 */
public final class StyleGenerator implements Function<Var, Var> {
    private final int latent;
    private final int width;
    private final GenerativeOps.Mlp mapping;
    private final Var constant;
    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();
    private final List<Var> styleW = new ArrayList<>();
    private final List<Var> styleB = new ArrayList<>();
    private final List<Var> noiseStrength = new ArrayList<>();
    private final SplittableRandom random;
    private float mixing = 0.5f;
    private boolean noise = true;
    private double pathMean;

    private StyleGenerator(int latent, int wDim, int width, int layers, int outputs, long seed) {
        this.latent = latent;
        this.width = width;
        this.random = new SplittableRandom(seed + 1);
        SplittableRandom r = new SplittableRandom(seed);
        mapping = new GenerativeOps.Mlp(r, false, latent, wDim, wDim, wDim);
        constant = TechniqueOps.normal(r, 1f, 1, width);
        for (int k = 0; k <= layers; k++) {
            int out = k == layers ? outputs : width;
            // fan-in scaled (demodulation makes the scale irrelevant to the hidden layers, but it
            // sets how fast Adam's fixed-size steps move them)
            weights.add(TechniqueOps.weight(r, width, out));
            biases.add(TechniqueOps.filled(0f, out));
            styleW.add(TechniqueOps.normal(r, (float) (1 / Math.sqrt(wDim)), wDim, width));
            styleB.add(TechniqueOps.filled(1f, width));
            if (k < layers) {
                noiseStrength.add(TechniqueOps.filled(0f, width));
            }
        }
    }

    /**
     * Creates the generator.
     *
     * @param latent the noise size
     * @param wDim the intermediate latent size
     * @param width the synthesis width
     * @param layers modulated hidden layers
     * @param outputs the output size
     * @param seed the initialization and noise seed
     * @return the generator
     */
    public static StyleGenerator of(int latent, int wDim, int width, int layers, int outputs, long seed) {
        return new StyleGenerator(latent, wDim, width, layers, outputs, seed);
    }

    /**
     * Sets the style-mixing probability during {@link #apply}.
     *
     * @param probability 0 to disable, 0.9 in StyleGAN
     * @return this
     */
    public StyleGenerator mixing(float probability) {
        this.mixing = probability;
        return this;
    }

    /**
     * Turns noise injection on or off.
     *
     * @param on whether to inject noise
     * @return this
     */
    public StyleGenerator noise(boolean on) {
        this.noise = on;
        return this;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(mapping.parameters());
        p.add(constant);
        p.addAll(weights);
        p.addAll(biases);
        p.addAll(styleW);
        p.addAll(styleB);
        p.addAll(noiseStrength);
        return p;
    }

    /**
     * The mapping network.
     *
     * @param z {@code [n, latent]}
     * @return {@code w}, {@code [n, wDim]}
     */
    public Var map(Var z) {
        Var norm = z.square().mean(1, true).add(1e-8f).sqrt();
        return mapping.apply(z.div(norm));
    }

    private static Var unit(Var w) {
        return w.div(w.square().mean(1, true).add(1e-8f).sqrt());
    }

    /**
     * The synthesis network.
     *
     * @param w the styles of the first {@code crossover} layers, {@code [n, wDim]}
     * @param w2 the styles of the rest, or {@code null}
     * @param crossover the first layer driven by {@code w2}
     * @param withNoise whether to inject noise
     * @return the tanh outputs {@code [n, outputs]}
     */
    public Var synthesize(Var w, Var w2, int crossover, boolean withNoise) {
        int n = w.shape().dim(0);
        // unit-RMS styles before the affine maps, so each style varies by about 1 around 1
        Var wa = unit(w);
        Var wb = w2 == null ? null : unit(w2);
        Var h = Var.of(Tensor.ones(Shape.of(n, 1))).matmul(constant);
        int layers = weights.size();
        for (int k = 0; k < layers; k++) {
            Var style = (wb != null && k >= crossover ? wb : wa).matmul(styleW.get(k)).add(styleB.get(k));
            Var wk = weights.get(k);
            Var y = h.mul(style).matmul(wk);
            if (k < layers - 1) {
                // demodulation: unit expected output scale for unit-scale inputs
                Var demod = style.square().matmul(wk.square()).add(1e-8f).sqrt();
                y = y.div(demod).add(biases.get(k));
                if (withNoise) {
                    y = y.add(Var.of(Tensor.normal(Shape.of(n, width), 0, 1, random)).mul(noiseStrength.get(k)));
                }
                h = GenerativeOps.silu(y);
            } else {
                h = y.add(biases.get(k));
            }
        }
        return h.tanh();
    }

    /**
     * Generates from noise, with style mixing and noise injection as configured.
     *
     * @param z {@code [n, latent]}, or {@code [n, 2 latent]} to allow mixing
     * @return {@code [n, outputs]}
     */
    @Override
    public Var apply(Var z) {
        Var w = map(z.slice(1, 0, latent));
        if (z.shape().dim(1) >= 2 * latent && random.nextDouble() < mixing) {
            Var w2 = map(z.slice(1, latent, 2 * latent));
            return synthesize(w, w2, 1 + random.nextInt(weights.size() - 1), noise);
        }
        return synthesize(w, null, weights.size(), noise);
    }

    /**
     * The path-length penalty on a batch of fresh latents; updates the running mean.
     *
     * @param n the batch size
     * @param rng the noise source
     * @return {@code mean (|J_w^T y| - a)^2}
     */
    public Var pathLengthPenalty(int n, RandomGenerator rng) {
        Var w = Var.param(map(Var.of(Tensor.normal(Shape.of(n, latent), 0, 1, rng))).value());
        Var img = synthesize(w, null, weights.size(), false);
        int outputs = img.shape().dim(1);
        Tensor y = Tensor.normal(img.shape(), 0, (float) (1 / Math.sqrt(outputs)), rng);
        Var g = Var.gradients(img.mul(y).sum(), List.of(w), true).get(0);
        Var lengths = g.square().sum(1, true).add(1e-8f).sqrt();
        double batchMean = lengths.value().mean().item();
        pathMean = pathMean == 0 ? batchMean : pathMean + 0.01 * (batchMean - pathMean);
        return lengths.sub((float) pathMean).square().mean();
    }
}
