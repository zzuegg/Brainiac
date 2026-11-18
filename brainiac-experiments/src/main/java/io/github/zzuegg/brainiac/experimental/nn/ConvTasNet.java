package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Conv-TasNet, time-domain source separation: a learned 1-D convolutional encoder
 * (windows of {@code L} samples at stride {@code L/2}, {@code N} filters, ReLU) replaces the
 * STFT; a layer norm, a bottleneck and a {@link TemporalConvNet} (dilated, causal) predict one sigmoid
 * mask per source over the encoder's {@code N} channels; a transposed-convolution decoder
 * (a linear map back to {@code L} samples per window, overlap-added) turns each masked
 * representation into a waveform. Input {@code [batch, samples]} (a multiple of {@code L/2}),
 * output {@code [batch, sources, samples]}. Train it with {@link PermutationInvariantLoss}.
 *
 * <p>Paper: Luo and Mesgarani, "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude
 * Masking for Speech Separation" (IEEE/ACM TASLP 2019),
 * <a href="https://arxiv.org/abs/1809.07454">arXiv:1809.07454</a>.
 */
public final class ConvTasNet implements Layer {

    private final int window;
    private final int filters;
    private final int sources;
    private final Var encoder;
    private final LayerNorm norm;
    private final Var bottleneck;
    private final TemporalConvNet tcn;
    private final Var maskProj;
    private final Var maskBias;
    private final Var decoder;

    /**
     * Creates the separator.
     *
     * @param window encoder window {@code L} in samples (even)
     * @param filters encoder filters {@code N}
     * @param channels TCN channels
     * @param levels TCN residual blocks (dilations 1, 2, 4, ...)
     * @param sources sources to separate
     * @param random the random source
     */
    public ConvTasNet(int window, int filters, int channels, int levels, int sources, SplittableRandom random) {
        this.window = window;
        this.filters = filters;
        this.sources = sources;
        float[][] init = analysisSynthesisPair(window, filters, random);
        encoder = Var.param(Tensor.of(Shape.of(window, filters), init[0]));
        norm = LayerNorm.of(filters);
        bottleneck = Var.param(Tensor.normal(Shape.of(filters, channels), 0, (float) Math.sqrt(1.0 / filters),
                random));
        tcn = new TemporalConvNet(channels, channels, 3, levels, random);
        maskProj = Var.param(Tensor.normal(Shape.of(channels, sources * filters), 0,
                (float) Math.sqrt(1.0 / channels), random));
        maskBias = Var.param(Tensor.zeros(Shape.of(sources * filters)));
        decoder = Var.param(Tensor.of(Shape.of(filters, window), init[1]));
    }

    /**
     * An encoder/decoder pair that starts as a perfect reconstruction: the first
     * {@code 2 window} filters are a random orthonormal basis {@code Q} and its negation, so
     * {@code relu(x Q) - relu(-x Q) = x Q}, and the decoder holds {@code Q^T} and {@code -Q^T}
     * halved for the two-fold overlap-add; further filters start small and random. With
     * masks of one the network then begins at the identity, not at a random filter (which
     * starts training around -20 dB SI-SNR).
     */
    private static float[][] analysisSynthesisPair(int window, int filters, SplittableRandom random) {
        double[][] q = new double[window][window];
        for (int i = 0; i < window; i++) {
            for (int j = 0; j < window; j++) {
                q[i][j] = random.nextGaussian();
            }
            for (int k = 0; k < i; k++) {
                double dot = 0;
                for (int j = 0; j < window; j++) {
                    dot += q[i][j] * q[k][j];
                }
                for (int j = 0; j < window; j++) {
                    q[i][j] -= dot * q[k][j];
                }
            }
            double norm = 0;
            for (int j = 0; j < window; j++) {
                norm += q[i][j] * q[i][j];
            }
            for (int j = 0; j < window; j++) {
                q[i][j] /= Math.sqrt(norm);
            }
        }
        float[] enc = new float[window * filters];
        float[] dec = new float[filters * window];
        for (int f = 0; f < filters; f++) {
            for (int j = 0; j < window; j++) {
                if (f < 2 * window) {
                    double sign = f < window ? 1 : -1;
                    double v = sign * q[f % window][j];
                    enc[j * filters + f] = (float) v;   // filter f is row f % window of Q
                    dec[f * window + j] = (float) (0.5 * v);
                } else {
                    enc[j * filters + f] = (float) (0.1 * random.nextGaussian() / Math.sqrt(window));
                    dec[f * window + j] = (float) (0.1 * random.nextGaussian() / Math.sqrt(filters));
                }
            }
        }
        return new float[][] {enc, dec};
    }

    /**
     * The encoder's representation.
     *
     * @param mixture {@code [b, samples]}
     * @return {@code [b, windows, N]}, windows = {@code samples / (L/2) - 1}
     */
    public Var encode(Var mixture) {
        int b = mixture.shape().dim(0);
        int stride = window / 2;
        int chunks = mixture.shape().dim(1) / stride;
        Var c = mixture.reshape(b, chunks, stride);
        Var frames = Var.concat(2, c.slice(1, 0, chunks - 1), c.slice(1, 1, chunks));
        return frames.matmul(encoder).relu();
    }

    /** Overlap-adds windows {@code [b, k, L]} at stride {@code L/2} to {@code [b, (k + 1) L/2]}. */
    private Var overlapAdd(Var d) {
        int b = d.shape().dim(0);
        int k = d.shape().dim(1);
        int stride = window / 2;
        Var zero = Var.of(Tensor.zeros(Shape.of(b, 1, stride)));
        Var first = Var.concat(1, d.slice(2, 0, stride), zero);
        Var second = Var.concat(1, zero, d.slice(2, stride, window));
        return first.add(second).reshape(b, (k + 1) * stride);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        int b = input.shape().dim(0);
        int samples = input.shape().dim(1);
        Var w = encode(input);
        Var masks = tcn.forward(norm.forward(w).matmul(bottleneck), training).matmul(maskProj).add(maskBias)
                .sigmoid();
        Var[] outs = new Var[sources];
        for (int i = 0; i < sources; i++) {
            Var m = masks.slice(2, i * filters, (i + 1) * filters);
            outs[i] = overlapAdd(w.mul(m).matmul(decoder)).reshape(b, 1, samples);
        }
        return Var.concat(1, outs);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(encoder));
        p.addAll(norm.parameters());
        p.add(bottleneck);
        p.addAll(tcn.parameters());
        p.addAll(List.of(maskProj, maskBias, decoder));
        return p;
    }

    @Override
    public String toString() {
        return "ConvTasNet(L=" + window + ", N=" + filters + ", sources=" + sources + ")";
    }
}
