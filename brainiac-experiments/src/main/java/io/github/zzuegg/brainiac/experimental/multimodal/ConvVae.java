package io.github.zzuegg.brainiac.experimental.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A small convolutional VAE for one-channel images, the latent space of
 * {@link TransfusionModel}'s latent-patch option (Transfusion encodes images with a pretrained
 * VAE): {@code side x side} pixels become a {@code channels x side/2 x side/2} latent (two 3x3
 * convolutions, the second with stride 2, then a 1x1 convolution to mean and log-variance);
 * the decoder mirrors it with nearest-neighbour upsampling. Training minimises the squared
 * error plus {@code beta} times the KL divergence. {@link #calibrate} sets one scale so the
 * latents have unit variance (the role of Stable Diffusion's 0.18215).
 *
 * <p>Latents as arrays are position-major: {@code [position][channel]}, one position per
 * latent pixel, row-major, which is the patch layout {@link TransfusionModel} expects.
 */
public final class ConvVae {

    private final int side;
    private final int channels;
    private final int hidden;
    private final float beta;
    private final Var e1;
    private final Var e1b;
    private final Var e2;
    private final Var e2b;
    private final Var e3;
    private final Var e3b;
    private final Var d1;
    private final Var d1b;
    private final Var d2;
    private final Var d2b;
    private final Var d3;
    private final Var d3b;
    private final SplittableRandom noise;
    private float scale = 1f;

    /**
     * Creates the VAE.
     *
     * @param side the image side, even
     * @param channels latent channels
     * @param hidden convolution channels
     * @param beta the KL weight
     * @param seed the seed
     */
    public ConvVae(int side, int channels, int hidden, float beta, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.side = side;
        this.channels = channels;
        this.hidden = hidden;
        this.beta = beta;
        this.e1 = conv(r, 1, hidden, 3);
        this.e1b = zeros(hidden);
        this.e2 = conv(r, hidden, hidden, 3);
        this.e2b = zeros(hidden);
        this.e3 = conv(r, hidden, 2 * channels, 1);
        this.e3b = zeros(2 * channels);
        this.d1 = conv(r, channels, hidden, 1);
        this.d1b = zeros(hidden);
        this.d2 = conv(r, hidden, hidden, 3);
        this.d2b = zeros(hidden);
        this.d3 = conv(r, hidden, 1, 3);
        this.d3b = zeros(1);
        this.noise = r.split();
    }

    static Var conv(SplittableRandom r, int in, int out, int k) {
        return Var.param(Tensor.normal(Shape.of(out, in, k, k), 0, (float) Math.sqrt(2.0 / (in * k * k)), r));
    }

    static Var zeros(int n) {
        return Var.param(Tensor.zeros(Shape.of(n)));
    }

    static Var silu(Var x) {
        return Activation.SILU.forward(x);
    }

    static Var apply(Var x, Var w, Var b, int stride) {
        int p = w.shape().dim(2) / 2;
        return Functional.conv(x, w, b, new int[] {stride, stride}, new int[] {p, p}, new int[] {p, p},
                new int[] {1, 1}, 1);
    }

    /** {mean, log-variance} of {@code [b, 1, side, side]} images, each {@code [b, channels, g, g]}. */
    private Var[] encode(Var images) {
        Var h = silu(apply(images, e1, e1b, 1));
        h = silu(apply(h, e2, e2b, 2));
        Var o = apply(h, e3, e3b, 1);
        return new Var[] {o.slice(1, 0, channels), o.slice(1, channels, 2 * channels)};
    }

    /** Images {@code [b, 1, side, side]} of latents {@code [b, channels, g, g]}. */
    private Var decode(Var z) {
        Var h = silu(apply(z, d1, d1b, 1));
        h = Functional.resize(h, new int[] {side, side}, Functional.Interpolation.NEAREST, false);
        h = silu(apply(h, d2, d2b, 1));
        return apply(h, d3, d3b, 1);
    }

    private Tensor images(float[][] x) {
        float[] flat = new float[x.length * side * side];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, flat, i * side * side, side * side);
        }
        return Tensor.of(Shape.of(x.length, 1, side, side), flat);
    }

    /**
     * One training step: reconstruction plus {@code beta} KL.
     *
     * @param batch images, {@code side * side} values each
     * @param optimizer the optimiser
     * @return the loss
     */
    public double trainStep(float[][] batch, Optimizer optimizer) {
        Var x = Var.of(images(batch));
        Var[] ml = encode(x);
        Tensor eps = Tensor.normal(ml[0].shape(), 0, 1f, noise);
        Var z = ml[0].add(ml[1].mul(0.5f).exp().mul(eps));
        Var rec = decode(z).sub(x).square().mean();
        Var kl = ml[0].square().add(ml[1].exp()).sub(ml[1]).sub(1f).mean().mul(0.5f);
        Var loss = rec.add(kl.mul(beta));
        double v = loss.value().item();
        loss.backward();
        optimizer.step(parameters());
        return v;
    }

    /**
     * Sets the latent scale so encoded means have unit standard deviation.
     *
     * @param data images
     */
    public void calibrate(float[][] data) {
        scale = 1f;
        float[][] z = encode(data);
        double sum = 0;
        double sq = 0;
        long n = 0;
        for (float[] row : z) {
            for (float v : row) {
                sum += v;
                sq += v * v;
                n++;
            }
        }
        double mean = sum / n;
        scale = (float) (1 / Math.sqrt(Math.max(1e-8, sq / n - mean * mean)));
    }

    /**
     * Scaled latent means, position-major.
     *
     * @param data images
     * @return per image {@code (side/2)^2 * channels} values
     */
    public float[][] encode(float[][] data) {
        int g = side / 2;
        float[] m = Var.noGrad(() -> encode(Var.of(images(data)))[0]).value().toArray();
        float[][] out = new float[data.length][g * g * channels];
        for (int i = 0; i < data.length; i++) {
            for (int c = 0; c < channels; c++) {
                for (int p = 0; p < g * g; p++) {
                    out[i][p * channels + c] = m[(i * channels + c) * g * g + p] * scale;
                }
            }
        }
        return out;
    }

    /**
     * Decodes scaled position-major latents.
     *
     * @param latents per image {@code (side/2)^2 * channels} values
     * @return images, {@code side * side} values each
     */
    public float[][] decode(float[][] latents) {
        int g = side / 2;
        float[] flat = new float[latents.length * channels * g * g];
        for (int i = 0; i < latents.length; i++) {
            for (int c = 0; c < channels; c++) {
                for (int p = 0; p < g * g; p++) {
                    flat[(i * channels + c) * g * g + p] = latents[i][p * channels + c] / scale;
                }
            }
        }
        float[] x = Var.noGrad(() -> decode(Var.of(Tensor.of(Shape.of(latents.length, channels, g, g), flat))))
                .value().toArray();
        float[][] out = new float[latents.length][side * side];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(x, i * side * side, out[i], 0, side * side);
        }
        return out;
    }

    /** @return latent channels */
    public int channels() {
        return channels;
    }

    /** @return latent positions per image */
    public int positions() {
        return (side / 2) * (side / 2);
    }

    /** @return multiply-adds of encoding plus decoding one image */
    public long macs() {
        long px = (long) side * side;
        long lat = px / 4;
        return px * 9 * hidden + lat * 9 * hidden * hidden + lat * hidden * 2 * channels
                + lat * channels * hidden + px * 9 * hidden * hidden + px * 9 * hidden;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return new ArrayList<>(List.of(e1, e1b, e2, e2b, e3, e3b, d1, d1b, d2, d2b, d3, d3b));
    }

    @Override
    public String toString() {
        return "ConvVae(" + side + "px -> " + channels + "x" + side / 2 + "x" + side / 2 + ")";
    }
}
