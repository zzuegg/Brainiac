package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ConvTasNet;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Dual-path RNN separation (Luo, Chen and Yoshioka, ICASSP 2020, <a href="https://arxiv.org/abs/1910.06379">arXiv:1910.06379</a>):
 * the Conv-TasNet encoder and decoder ({@link ConvTasNet#analysisSynthesisPair},
 * {@link ConvTasNet#overlapAdd}) around a different mask network. The encoded frames
 * (layer-normalised, projected to {@code channels}) are split into chunks of {@code chunk}
 * frames with 50% overlap, giving a {@code [batch, chunks, chunk, channels]} block; each
 * dual-path block runs a bidirectional GRU <em>within</em> every chunk (local structure),
 * then one <em>across</em> chunks at every intra-chunk position (global structure), each
 * followed by a linear map, layer norm and a residual connection, so every frame sees the
 * whole utterance through recurrences of length about {@code sqrt(2 frames)}. The chunks are
 * overlap-added back, and a sigmoid mask per source multiplies the encoded mixture before
 * decoding. Input {@code [batch, samples]}, output {@code [batch, sources, samples]}.
 */
public final class DualPathRnn implements Layer {

    private final int window;
    private final int filters;
    private final int channels;
    private final int chunk;
    private final int sources;
    private final int samples;
    private final int frames;
    private final int chunks;
    private final Var encoder;
    private final LayerNorm norm;
    private final Var bottleneck;
    private final List<Layer> intra = new ArrayList<>();
    private final List<Var[]> intraOut = new ArrayList<>();
    private final List<LayerNorm> intraNorm = new ArrayList<>();
    private final List<Layer> inter = new ArrayList<>();
    private final List<Var[]> interOut = new ArrayList<>();
    private final List<LayerNorm> interNorm = new ArrayList<>();
    private final Var maskProj;
    private final Var maskBias;
    private final Var decoder;

    /**
     * Creates the separator.
     *
     * @param window the encoder window in samples (even; stride {@code window / 2})
     * @param filters encoder filters
     * @param channels the dual-path width (even)
     * @param chunk frames per chunk (even)
     * @param blocks dual-path blocks
     * @param sources sources to separate
     * @param samples the input length (a multiple of {@code window / 2})
     * @param random the initialisation source
     */
    public DualPathRnn(int window, int filters, int channels, int chunk, int blocks, int sources, int samples,
            SplittableRandom random) {
        this.window = window;
        this.filters = filters;
        this.channels = channels;
        this.chunk = chunk;
        this.sources = sources;
        this.samples = samples;
        this.frames = samples / (window / 2) - 1;
        int half = chunk / 2;
        this.chunks = (frames + half - 1) / half;          // padded length half * (chunks + 1) >= frames + half
        float[][] init = ConvTasNet.analysisSynthesisPair(window, filters, random);
        encoder = Var.param(Tensor.of(Shape.of(window, filters), init[0]));
        decoder = Var.param(Tensor.of(Shape.of(filters, window), init[1]));
        norm = LayerNorm.of(filters);
        bottleneck = Var.param(Tensor.normal(Shape.of(filters, channels), 0, (float) Math.sqrt(1.0 / filters),
                random));
        for (int i = 0; i < blocks; i++) {
            intra.add(Recurrent.gru(channels / 2).returnSequences(true).bidirectional(Recurrent.Merge.CONCAT)
                    .create(Shape.of(chunk, channels), random));
            intraOut.add(linear(channels, random));
            intraNorm.add(LayerNorm.of(channels));
            inter.add(Recurrent.gru(channels / 2).returnSequences(true).bidirectional(Recurrent.Merge.CONCAT)
                    .create(Shape.of(chunks, channels), random));
            interOut.add(linear(channels, random));
            interNorm.add(LayerNorm.of(channels));
        }
        maskProj = Var.param(Tensor.normal(Shape.of(channels, sources * filters), 0,
                (float) Math.sqrt(1.0 / channels), random));
        maskBias = Var.param(Tensor.zeros(Shape.of(sources * filters)));
    }

    private static Var[] linear(int n, SplittableRandom random) {
        return new Var[] {Var.param(Tensor.normal(Shape.of(n, n), 0, (float) Math.sqrt(1.0 / n), random)),
            Var.param(Tensor.zeros(Shape.of(n)))};
    }

    /**
     * Encodes waveforms as Conv-TasNet does.
     *
     * @param mixture {@code [batch, samples]}
     * @return {@code [batch, frames, filters]}
     */
    public Var encode(Var mixture) {
        int b = mixture.shape().dim(0);
        int stride = window / 2;
        int k = mixture.shape().dim(1) / stride;
        Var c = mixture.reshape(b, k, stride);
        return Var.concat(2, c.slice(1, 0, k - 1), c.slice(1, 1, k)).matmul(encoder).relu();
    }

    /** Splits {@code [b, frames, C]} into {@code [b, chunks, chunk, C]} with 50% overlap. */
    private Var segment(Var x) {
        int b = x.shape().dim(0);
        int half = chunk / 2;
        int padded = half * (chunks + 1);
        Var p = Var.concat(1, x, Var.of(Tensor.zeros(Shape.of(b, padded - frames, channels))));
        Var halves = p.reshape(b, chunks + 1, half, channels);
        return Var.concat(2, halves.slice(1, 0, chunks), halves.slice(1, 1, chunks + 1));
    }

    /** Overlap-adds {@code [b, chunks, chunk, C]} back to {@code [b, frames, C]}. */
    private Var merge(Var x) {
        int b = x.shape().dim(0);
        int half = chunk / 2;
        Var zero = Var.of(Tensor.zeros(Shape.of(b, 1, half, channels)));
        Var first = Var.concat(1, x.slice(2, 0, half), zero);
        Var second = Var.concat(1, zero, x.slice(2, half, chunk));
        return first.add(second).reshape(b, (chunks + 1) * half, channels).slice(1, 0, frames);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() != 2 || input.shape().dim(1) != samples) {
            throw new IllegalArgumentException("DualPathRnn expects [batch, " + samples + "], got " + input.shape());
        }
        int b = input.shape().dim(0);
        Var w = encode(input);                                               // [b, frames, N]
        Var h = segment(norm.forward(w).matmul(bottleneck));                  // [b, chunks, chunk, C]
        for (int i = 0; i < intra.size(); i++) {
            Var a = intra.get(i).forward(h.reshape(b * chunks, chunk, channels), training);
            a = intraNorm.get(i).forward(a.matmul(intraOut.get(i)[0]).add(intraOut.get(i)[1]));
            h = h.add(a.reshape(b, chunks, chunk, channels));
            Var t = h.permute(0, 2, 1, 3).reshape(b * chunk, chunks, channels);
            Var e = inter.get(i).forward(t, training);
            e = interNorm.get(i).forward(e.matmul(interOut.get(i)[0]).add(interOut.get(i)[1]));
            h = h.add(e.reshape(b, chunk, chunks, channels).permute(0, 2, 1, 3));
        }
        Var masks = merge(h).relu().matmul(maskProj).add(maskBias).sigmoid();   // [b, frames, sources * N]
        Var[] outs = new Var[sources];
        for (int s = 0; s < sources; s++) {
            Var m = masks.slice(2, s * filters, (s + 1) * filters);
            outs[s] = ConvTasNet.overlapAdd(w.mul(m).matmul(decoder), window).reshape(b, 1, samples);
        }
        return Var.concat(1, outs);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(encoder));
        p.addAll(norm.parameters());
        p.add(bottleneck);
        for (int i = 0; i < intra.size(); i++) {
            p.addAll(intra.get(i).parameters());
            p.addAll(List.of(intraOut.get(i)));
            p.addAll(intraNorm.get(i).parameters());
            p.addAll(inter.get(i).parameters());
            p.addAll(List.of(interOut.get(i)));
            p.addAll(interNorm.get(i).parameters());
        }
        p.addAll(List.of(maskProj, maskBias, decoder));
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(sources, samples);
    }

    @Override
    public String toString() {
        return "DualPathRnn(L=" + window + ", N=" + filters + ", C=" + channels + ", chunk=" + chunk + ", chunks="
                + chunks + ")";
    }
}
