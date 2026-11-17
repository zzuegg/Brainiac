package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Landmark attention (Mohtashami and Jaggi 2023, <a href="https://arxiv.org/abs/2305.16300">arXiv:2305.16300</a>):
 * causal attention where the sequence is cut into blocks of {@code L} tokens and every
 * finished block is represented by a landmark key. A query takes one softmax over the
 * tokens of its own block (causally) and the landmarks of earlier blocks; the mass a
 * landmark gets is shared among its block's tokens by a softmax within the block (the
 * paper's grouped softmax). Because blocks are reached only through their landmarks, at
 * inference a query can keep only its {@link #retrieve top-k} landmarks and read just those
 * blocks, so a model trained on short contexts reads long ones by retrieval.
 *
 * <p>Here the landmark key of a block is a learned projection of the block's mean input
 * (the paper inserts a learned landmark token instead, whose key the layer computes the
 * same way as any other). Input {@code [batch, steps, features]}, output the same shape.
 * The within-block softmax is normalised against the row maximum over earlier blocks, so a
 * block whose scores are some 80 below that maximum underflows to no weight.
 */
public final class LandmarkAttention implements Layer {

    private static final float MASKED = -1e9f;

    private final int features;
    private final int heads;
    private final int headSize;
    private final int block;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wl;
    private final Var wo;
    private int topK;

    /**
     * Creates the layer.
     *
     * @param features the input and output features
     * @param heads the heads
     * @param headSize the size per head
     * @param block the tokens per block {@code L}
     * @param random the random source for the weights
     */
    public LandmarkAttention(int features, int heads, int headSize, int block, SplittableRandom random) {
        if (features < 1 || heads < 1 || headSize < 1 || block < 1) {
            throw new IllegalArgumentException("sizes must be positive");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = headSize;
        this.block = block;
        int hd = heads * headSize;
        wq = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wk = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wv = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wl = Var.param(Init.XAVIER.create(Shape.of(features, hd), features, hd, random));
        wo = Var.param(Init.XAVIER.create(Shape.of(hd, features), hd, features, random));
    }

    /**
     * Describes the layer for model builders, on inputs {@code [steps, features]}.
     *
     * @param heads the heads
     * @param headSize the size per head
     * @param block the tokens per block
     * @return the spec
     */
    public static LayerSpec spec(int heads, int headSize, int block) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new LandmarkAttention(input.dim(-1), heads, headSize, block, random);
            }
        };
    }

    /**
     * @param k the landmarks (earlier blocks) each query keeps, or 0 for all of them
     * @return this layer
     */
    public LandmarkAttention retrieve(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("k must be non-negative");
        }
        this.topK = k;
        return this;
    }

    private Var split(Var x, Var w, int n, int t) {
        return x.matmul(w).reshape(n, t, heads, headSize).permute(0, 2, 1, 3);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != features) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + features + "], got " + s);
        }
        int n = s.dim(0);
        int steps = s.dim(1);
        int blocks = Math.ceilDiv(steps, block);
        Device device = input.value().device();
        float[] current = new float[steps * steps];
        float[] earlier = new float[steps * steps];
        float[] landmark = new float[steps * blocks];
        float[] member = new float[steps * blocks];
        float[] mean = new float[steps * blocks];
        for (int i = 0; i < steps; i++) {
            int bi = i / block;
            for (int j = 0; j < steps; j++) {
                int bj = j / block;
                current[i * steps + j] = bj == bi && j <= i ? 0 : MASKED;
                earlier[i * steps + j] = bj < bi ? 0 : MASKED;
            }
            for (int b = 0; b < blocks; b++) {
                landmark[i * blocks + b] = b < bi ? 0 : MASKED;
            }
            member[i * blocks + bi] = 1;
            int size = Math.min(steps, (bi + 1) * block) - bi * block;
            mean[i * blocks + bi] = 1f / size;
        }
        Tensor ind = Tensor.of(Shape.of(steps, blocks), member, device);
        float scale = (float) (1 / Math.sqrt(headSize));
        Var q = split(input, wq, n, steps);
        Var k = split(input, wk, n, steps);
        Var v = split(input, wv, n, steps);
        Var means = input.transpose().matmul(Tensor.of(Shape.of(steps, blocks), mean, device)).transpose();
        Var l = split(means, wl, n, blocks);
        Var scores = q.matmul(k.transpose()).mul(scale);
        Var marks = q.matmul(l.transpose()).mul(scale);
        Tensor landmarkMask = Tensor.of(Shape.of(steps, blocks), landmark, device);
        if (topK > 0) {
            landmarkMask = retrievalMask(marks.value(), landmark, n, steps, blocks, device);
        }
        Var top = Activation.SOFTMAX.forward(Var.concat(3,
                scores.add(Tensor.of(Shape.of(steps, steps), current, device)), marks.add(landmarkMask)));
        Var direct = top.slice(3, 0, steps);
        Var viaLandmark = top.slice(3, steps, steps + blocks);
        Var inBlock = scores.add(Tensor.of(Shape.of(steps, steps), earlier, device));
        Var e = inBlock.sub(inBlock.max(3, true).detach()).exp();
        Var blockSums = e.matmul(ind).matmul(ind.transpose());
        Var share = e.div(blockSums.add(1e-12f));
        Var weights = direct.add(viaLandmark.matmul(ind.transpose()).mul(share));
        return weights.matmul(v).permute(0, 2, 1, 3).reshape(n, steps, heads * headSize).matmul(wo);
    }

    /** Keeps each query's {@code topK} best allowed landmarks. */
    private Tensor retrievalMask(Tensor marks, float[] allowed, int n, int steps, int blocks, Device device) {
        float[] g = marks.toArray();
        float[] mask = new float[g.length];
        Arrays.fill(mask, MASKED);
        Integer[] order = new Integer[blocks];
        for (int row = 0; row < n * heads * steps; row++) {
            int i = row % steps;
            int base = row * blocks;
            for (int b = 0; b < blocks; b++) {
                order[b] = b;
            }
            Arrays.sort(order, (a, c) -> Float.compare(g[base + c], g[base + a]));
            int kept = 0;
            for (int b : order) {
                if (kept < topK && allowed[i * blocks + b] == 0) {
                    mask[base + b] = 0;
                    kept++;
                }
            }
        }
        return Tensor.of(Shape.of(n, heads, steps, blocks), mask, device);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wl, wo);
    }

    @Override
    public String toString() {
        return "LandmarkAttention(" + heads + " heads, block " + block + (topK > 0 ? ", top " + topK : "") + ")";
    }
}
