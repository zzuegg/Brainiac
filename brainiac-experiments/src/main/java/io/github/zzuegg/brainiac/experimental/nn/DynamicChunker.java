package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * H-Net dynamic chunking: a learned, end-to-end tokenizer inside the network. Byte (or
 * token) states from an encoder go in; the layer decides where chunks begin, runs a main
 * network once per chunk, and hands every byte the output of its chunk:
 *
 * <ol>
 *   <li><b>routing</b>: a boundary probability from the cosine similarity of adjacent
 *       states, {@code p_t = (1 - cos(x_t W_q, x_{t-1} W_k)) / 2} ({@code p_0 = 1}); a chunk
 *       starts where {@code p_t >= 1/2}, so a change of content starts a new chunk;</li>
 *   <li><b>downsampling</b>: the states at the boundaries form the chunk sequence, which the
 *       main network ({@code LayerSpec}, causal) processes;</li>
 *   <li><b>smoothing</b> (dechunking): an exponential moving average over chunks,
 *       {@code z~_c = P_c z_c + (1 - P_c) z~_{c-1}} with {@code P_c} the boundary probability,
 *       which makes the discrete choice differentiable and blurs uncertain boundaries;</li>
 *   <li><b>upsampling</b>: every byte takes its chunk's smoothed state, scaled by a confidence
 *       {@code c_t} ({@code p_t} at a boundary, {@code 1 - p_t} inside) with a
 *       straight-through estimator (value 1, gradient of {@code c_t}), plus a residual
 *       {@code x_t W_res} from the encoder.</li>
 * </ol>
 *
 * <p>A ratio loss steers the compression towards a target of {@code N} bytes per chunk:
 * {@code N/(N-1) ((N-1) F G + (1-F)(1-G))} with {@code F} the share of boundaries and
 * {@code G} the mean boundary probability; it is the {@link #auxiliaryLoss()}, times
 * {@code ratioWeight}. Causal: a byte only sees chunks that start at or before it.
 *
 * <p>Input and output are {@code [batch, steps, features]}; sequences are chunked one at a
 * time (their chunk counts differ).
 *
 * <p>Paper: Hwang et al., "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
 * <a href="https://arxiv.org/abs/2507.07955">arXiv:2507.07955</a>.
 */
public final class DynamicChunker implements Layer {

    /**
     * One sequence's chunking.
     *
     * @param probabilities the boundary probability of every step {@code [steps]}
     * @param starts the steps where chunks begin, from 0
     * @param chunkOf the chunk of every step
     */
    public record Chunks(Var probabilities, int[] starts, int[] chunkOf) {

        /** @return the number of chunks */
        public int count() {
            return starts.length;
        }
    }

    private final int features;
    private final Layer main;
    private final float targetRatio;
    private final float ratioWeight;
    private final Var wq;
    private final Var wk;
    private final Var wRes;
    private Var lastRatioLoss;
    private double lastBytesPerChunk;

    /**
     * Creates the layer.
     *
     * @param features the width of byte and chunk states
     * @param main the main network over chunks, {@code [1, chunks, features]} to the same shape
     * @param targetRatio the target bytes per chunk {@code N}, above 1
     * @param ratioWeight the weight of the ratio loss
     * @param random the random source
     */
    public DynamicChunker(int features, Layer main, float targetRatio, float ratioWeight, SplittableRandom random) {
        if (targetRatio <= 1) {
            throw new IllegalArgumentException("the target ratio must exceed 1");
        }
        this.features = features;
        this.main = main;
        this.targetRatio = targetRatio;
        this.ratioWeight = ratioWeight;
        this.wq = identityPlusNoise(features, random);
        this.wk = identityPlusNoise(features, random);
        this.wRes = identityPlusNoise(features, random);
    }

    private static Var identityPlusNoise(int n, SplittableRandom random) {
        float[] w = Tensor.uniform(Shape.of(n, n), -0.02f, 0.02f, random).toArray();
        for (int i = 0; i < n; i++) {
            w[i * n + i] += 1;
        }
        return Var.param(Tensor.of(Shape.of(n, n), w));
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, ratio-loss weight 0.03.
     *
     * @param main the main network spec (built for the input shape)
     * @param targetRatio the target bytes per chunk
     * @return the spec
     */
    public static LayerSpec of(LayerSpec main, float targetRatio) {
        return MixerOps.sameShape((in, r) -> new DynamicChunker(in.dim(-1), main.create(in, r), targetRatio, 0.03f,
                r));
    }

    /**
     * Routes one sequence.
     *
     * @param x byte states {@code [steps, features]}
     * @return the chunking
     */
    public Chunks route(Var x) {
        int t = x.shape().dim(0);
        Var q = MixerOps.unit(x.matmul(wq));
        Var k = MixerOps.unit(x.matmul(wk));
        Var p;
        if (t == 1) {
            p = Var.of(Tensor.full(Shape.of(1), 1f));
        } else {
            Var cos = q.slice(0, 1, t).mul(k.slice(0, 0, t - 1)).sum(-1, false);    // [t - 1]
            p = Var.concat(0, Var.of(Tensor.full(Shape.of(1), 1f)), cos.neg().add(1f).mul(0.5f).clamp(0f, 1f));
        }
        float[] pv = p.value().toArray();
        List<Integer> starts = new ArrayList<>();
        int[] chunkOf = new int[t];
        for (int i = 0; i < t; i++) {
            if (i == 0 || pv[i] >= 0.5f) {
                starts.add(i);
            }
            chunkOf[i] = starts.size() - 1;
        }
        return new Chunks(p, starts.stream().mapToInt(Integer::intValue).toArray(), chunkOf);
    }

    /**
     * Smooths chunk outputs and hands them back to the bytes.
     *
     * @param z the main network's chunk outputs {@code [chunks, features]}
     * @param chunks the chunking
     * @return {@code [steps, features]}, before the residual
     */
    public Var dechunk(Var z, Chunks chunks) {
        int n = chunks.count();
        Var pc = chunks.probabilities().gather(0, MixerOps.ids(chunks.starts())).reshape(n, 1);
        List<Var> smoothed = new ArrayList<>();
        Var prev = null;
        for (int c = 0; c < n; c++) {
            Var pcc = pc.slice(0, c, c + 1);
            Var zc = z.slice(0, c, c + 1);
            prev = prev == null ? zc : zc.mul(pcc).add(prev.mul(pcc.neg().add(1f)));
            smoothed.add(prev);
        }
        Var zs = n == 1 ? smoothed.getFirst() : Var.concat(0, smoothed.toArray(Var[]::new));
        Var up = zs.gather(0, MixerOps.ids(chunks.chunkOf()));                   // [t, features]
        int t = chunks.chunkOf().length;
        float[] isStart = new float[t];
        for (int s : chunks.starts()) {
            isStart[s] = 1;
        }
        Tensor b = Tensor.of(Shape.of(t), isStart);
        Var p = chunks.probabilities();
        Var confidence = p.mul(b).add(p.neg().add(1f).mul(b.neg().add(1f)));     // [t]
        Var ste = Functional.straightThrough(Var.of(Tensor.full(Shape.of(t), 1f)), confidence);
        return up.mul(ste.reshape(t, 1));
    }

    /**
     * The ratio loss of one chunking.
     *
     * @param chunks the chunking
     * @return the scalar loss, 1 at the target ratio
     */
    public Var ratioLoss(Chunks chunks) {
        int t = chunks.chunkOf().length;
        float f = chunks.count() / (float) t;
        Var g = chunks.probabilities().mean();
        float n = targetRatio;
        return g.mul((n - 1) * f).add(g.neg().add(1f).mul(1 - f)).mul(n / (n - 1));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        MixerOps.requireSequence(input, features, "DynamicChunker");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        List<Var> outs = new ArrayList<>();
        Var ratio = null;
        long chunkTotal = 0;
        for (int n = 0; n < b; n++) {
            Var x = input.slice(0, n, n + 1).reshape(t, features);
            Chunks c = route(x);
            chunkTotal += c.count();
            Var selected = x.gather(0, MixerOps.ids(c.starts())).reshape(1, c.count(), features);
            Var z = main.forward(selected, training).reshape(c.count(), features);
            outs.add(dechunk(z, c).add(x.matmul(wRes)).reshape(1, t, features));
            Var r = ratioLoss(c);
            ratio = ratio == null ? r : ratio.add(r);
        }
        lastBytesPerChunk = b * t / (double) chunkTotal;
        lastRatioLoss = training ? ratio.div(b).mul(ratioWeight) : null;
        return b == 1 ? outs.getFirst() : Var.concat(0, outs.toArray(Var[]::new));
    }

    /** @return the weighted ratio loss of the last training pass, plus the main network's own */
    @Override
    public Var auxiliaryLoss() {
        Var a = main.auxiliaryLoss();
        if (lastRatioLoss == null) {
            return a;
        }
        return a == null ? lastRatioLoss : lastRatioLoss.add(a);
    }

    /** @return the mean number of bytes per chunk in the last pass (the compression ratio) */
    public double lastBytesPerChunk() {
        return lastBytesPerChunk;
    }

    /** @return the main network */
    public Layer main() {
        return main;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(main.parameters());
        p.addAll(List.of(wq, wk, wRes));
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "DynamicChunker(" + features + ", target ratio " + targetRatio + ", main " + main + ")";
    }
}
