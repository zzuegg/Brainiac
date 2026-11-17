package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * VQ-GAN (Esser, Rombach and Ommer, <a href="https://arxiv.org/abs/2012.09841">arXiv:2012.09841</a>):
 * a spatial vector-quantised autoencoder for single-channel images trained, beyond the VQ-VAE
 * objective (reconstruction, codebook and commitment terms of {@link VectorQuantizer}), with a
 * <b>perceptual</b> loss (distance between a frozen feature network's features of the input
 * and of the reconstruction) and a <b>patch-adversarial</b> hinge loss from a
 * {@link PatchDiscriminator}, switched on after a warm-up, so the decoder commits to sharp,
 * realistic detail even when a small codebook cannot store it exactly.
 *
 * <p>The encoder is a 3x3 convolution and a stride-2 4x4 convolution, so an
 * {@code side x side} image becomes {@code (side / 2)^2} tokens of {@code dim} channels, each
 * quantised to the nearest of {@code codes} codebook vectors; the decoder mirrors it with a
 * transposed convolution. With both extra weights at zero it is the VQ-VAE baseline.
 */
public final class VqGan {
    private static final int[] ONE = {1, 1};
    private final int side;
    private final int dim;
    private final VectorQuantizer quantizer;
    private final List<Var> encoder = new ArrayList<>();
    private final List<Var> decoder = new ArrayList<>();
    private Function<Var, Var> features;
    private float perceptualWeight;
    private PatchDiscriminator discriminator;
    private float adversarialWeight;
    private int adversarialStart;
    private int steps;

    private VqGan(int side, int codes, int dim, int width, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.side = side;
        this.dim = dim;
        conv(encoder, r, width, 1, 3);
        conv(encoder, r, dim, width, 4);
        // transposed convolution weights are [in, out, k, k]
        decoder.add(TechniqueOps.normal(r, (float) Math.sqrt(2.0 / (dim * 16)), dim, width, 4, 4));
        decoder.add(TechniqueOps.filled(0f, width));
        conv(decoder, r, width, width, 3);
        conv(decoder, r, 1, width, 3);
        quantizer = VectorQuantizer.of(codes, dim, 0.25f, seed + 1);
    }

    private static void conv(List<Var> into, SplittableRandom r, int out, int in, int k) {
        into.add(TechniqueOps.normal(r, (float) Math.sqrt(2.0 / (in * k * k)), out, in, k, k));
        into.add(TechniqueOps.filled(0f, out));
    }

    /**
     * Creates the autoencoder.
     *
     * @param side the image side, even
     * @param codes the codebook size
     * @param dim the code vector size
     * @param width the convolution channels
     * @param seed the initialization seed
     * @return the model, a plain VQ-VAE until {@link #perceptual} or {@link #adversarial} is set
     */
    public static VqGan of(int side, int codes, int dim, int width, long seed) {
        return new VqGan(side, codes, dim, width, seed);
    }

    /**
     * Adds the perceptual loss.
     *
     * @param features a frozen feature network on flat images {@code [n, side * side]}
     * @param weight its weight
     * @return this
     */
    public VqGan perceptual(Function<Var, Var> features, float weight) {
        this.features = features;
        this.perceptualWeight = weight;
        return this;
    }

    /**
     * Adds the patch-adversarial loss.
     *
     * @param discriminator the patch discriminator on {@code [n, 1, side, side]}
     * @param weight the generator's adversarial weight
     * @param start the training step from which it is used
     * @return this
     */
    public VqGan adversarial(PatchDiscriminator discriminator, float weight, int start) {
        this.discriminator = discriminator;
        this.adversarialWeight = weight;
        this.adversarialStart = start;
        return this;
    }

    /** @return the autoencoder's parameters (codebook included, discriminator excluded) */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(encoder);
        p.addAll(decoder);
        p.addAll(quantizer.parameters());
        return p;
    }

    /** @return the quantizer */
    public VectorQuantizer quantizer() {
        return quantizer;
    }

    private static Var silu(Var x) {
        return x.mul(x.sigmoid());
    }

    private Var c(Var x, List<Var> p, int i, int stride) {
        return Functional.conv(x, p.get(2 * i), p.get(2 * i + 1), new int[] {stride, stride}, ONE, ONE, ONE, 1);
    }

    /** Encoder output as rows {@code [n * tokens, dim]}. */
    private Var encode(Var flat) {
        int n = flat.shape().dim(0);
        Var h = silu(c(flat.reshape(n, 1, side, side), encoder, 0, 1));
        Var z = c(h, encoder, 1, 2);
        int t = side / 2;
        return z.permute(0, 2, 3, 1).reshape(n * t * t, dim);
    }

    private Var decode(Var rows, int n) {
        int t = side / 2;
        Var z = rows.reshape(n, t, t, dim).permute(0, 3, 1, 2);
        Var h = silu(Functional.convTranspose(z, decoder.get(0), decoder.get(1), new int[] {2, 2}, ONE, ONE,
                new int[] {0, 0}, ONE, 1));
        h = silu(c(h, decoder, 1, 1));
        return c(h, decoder, 2, 1).tanh().reshape(n, side * side);
    }

    /**
     * Reconstructs images through the codebook.
     *
     * @param images flat {@code [n, side * side]} in [-1, 1]
     * @return reconstructions
     */
    public Tensor reconstruct(Tensor images) {
        int n = images.shape().dim(0);
        return decode(quantizer.lookup(quantizer.nearest(encode(Var.of(images)).value())), n).value();
    }

    /**
     * The code of every token.
     *
     * @param images flat images
     * @return {@code n * (side / 2)^2} code ids, image by image
     */
    public int[] codes(Tensor images) {
        return quantizer.nearest(encode(Var.of(images)).value());
    }

    /**
     * Re-seeds unused codes from the encoder outputs of a batch.
     *
     * @param images flat images
     * @return the number of codes restarted
     */
    public int restart(Tensor images) {
        return quantizer.restartDeadCodes(encode(Var.of(images)).value());
    }

    /**
     * One training step: the discriminator first (once the adversarial loss is active), then
     * the autoencoder.
     *
     * @param images flat {@code [n, side * side]} in [-1, 1]
     * @param optimizer the autoencoder's optimizer
     * @param discriminatorOptimizer the discriminator's, or {@code null} without one
     * @return the pixel reconstruction loss (mean squared error)
     */
    public float step(Tensor images, Optimizer optimizer, Optimizer discriminatorOptimizer) {
        int n = images.shape().dim(0);
        Var x = Var.of(images);
        VectorQuantizer.Result q = quantizer.quantize(encode(x));
        Var recon = decode(q.output(), n);
        Var rec = recon.sub(x).square().mean();
        Var loss = rec.add(q.loss());
        if (features != null && perceptualWeight > 0) {
            loss = loss.add(features.apply(recon).sub(features.apply(x).detach()).square().mean()
                    .mul(perceptualWeight));
        }
        boolean adversarial = discriminator != null && steps >= adversarialStart;
        if (adversarial) {
            Var real = discriminator.forward(x.reshape(n, 1, side, side));
            Var fake = discriminator.forward(recon.detach().reshape(n, 1, side, side));
            Var d = real.neg().add(1f).relu().mean().add(fake.add(1f).relu().mean());
            d.backward();
            for (Var p : parameters()) {
                p.zeroGrad();
            }
            discriminatorOptimizer.step(discriminator.parameters());
            loss = loss.add(discriminator.forward(recon.reshape(n, 1, side, side)).mean().neg()
                    .mul(adversarialWeight));
        }
        loss.backward();
        if (discriminator != null) {
            for (Var p : discriminator.parameters()) {
                p.zeroGrad();
            }
        }
        optimizer.step(parameters());
        steps++;
        return rec.value().item();
    }

    /**
     * Initialises the codebook from encoder outputs (k-means++).
     *
     * @param images flat images
     */
    public void initialize(Tensor images) {
        quantizer.initialize(encode(Var.of(images)).value());
    }

    /** @return the code vector size */
    public int dim() {
        return dim;
    }
}
