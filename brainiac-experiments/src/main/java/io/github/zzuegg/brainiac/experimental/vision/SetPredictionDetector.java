package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Boxes;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * DETR-lite: object detection as direct set prediction (Carion et al., 2020), with optional
 * denoising queries (DN-DETR, Li et al., 2022). A {@link ConvBackbone} map (stride 4) plus
 * learned positions forms the memory; {@code N} learned object queries pass through a small
 * transformer decoder (self-attention among queries, cross-attention to the memory, MLP) and
 * each emits a class distribution over the classes plus "no object" and a normalised box.
 *
 * <p>The loss first finds the optimal one-to-one assignment between predictions and ground
 * truth with the Hungarian algorithm ({@link MinibatchOptimalTransport#assignCost}) on a
 * cost of {@code -p(class) + 5 L1 + 2 (1 - GIoU)}, then trains every query on its assignment
 * (unmatched ones on "no object", weighted 0.1), so duplicates are penalised and no NMS is
 * needed. With denoising, each image's ground-truth boxes, jittered (centre up to 40% of the
 * size, size up to 40%) and with 20% of labels flipped, are fed as extra queries that must
 * reconstruct the clean box and label; an attention mask stops the matching queries from
 * seeing them. Those queries have a fixed target, so they give the decoder a stable
 * box-refinement signal while the matching is still erratic.
 *
 * <p>As a {@link Layer} it maps images to {@code [batch, N, classes + 1 + 4]}.
 *
 * <p>Papers: Carion et al., "End-to-End Object Detection with Transformers" (2020),
 * <a href="https://arxiv.org/abs/2005.12872">arXiv:2005.12872</a>; Li et al., "DN-DETR:
 * Accelerate DETR Training by Introducing Query DeNoising" (2022),
 * <a href="https://arxiv.org/abs/2203.01305">arXiv:2203.01305</a>.
 */
public final class SetPredictionDetector implements Layer {

    /**
     * Convergence options, all off by default (the wave 8 DETR-lite).
     *
     * <ul>
     *   <li>{@code hybridQueries > 0} adds H-DETR's one-to-many branch (Jia et al., CVPR
     *       2023): a second group of queries, hidden from the one-to-one queries by an
     *       attention mask, trained with every ground-truth box repeated
     *       {@code hybridRepeat} times in the Hungarian matching, which gives many more
     *       positive gradients; inference uses only the one-to-one queries, so it stays
     *       NMS-free.</li>
     *   <li>{@code anchorQueries} makes the queries explicit 4-D anchor boxes (DAB-DETR, Liu
     *       et al., ICLR 2022): a sine embedding of each anchor through an MLP is the
     *       positional query added to the attention queries and keys, and every layer
     *       predicts a correction in logit space that refines the anchor for the next
     *       layer (the anchor is detached between layers).</li>
     *   <li>{@code auxiliaryLosses} applies the matching loss to every decoder layer's
     *       output, not only the last.</li>
     *   <li>{@code querySelection} is two-stage query selection (two-stage Deformable DETR, Zhu
     *       et al., ICLR 2021; RT-DETR, Zhao et al., CVPR 2024): every memory token predicts a
     *       class and a box (relative to its cell) with encoder heads trained by their own
     *       Hungarian loss, and the top-{@code queries} tokens by best real-class score become
     *       the decoder queries: their features are the query contents and their boxes
     *       (detached) the reference anchors, refined layer by layer as with
     *       {@code anchorQueries}. The decoder therefore starts near the objects instead of from
     *       learned constants. Hybrid queries, if any, are selected the same way.</li>
     * </ul>
     *
     * <p>Options cannot be combined with denoising queries.
     *
     * @param hybridQueries the one-to-many queries, 0 for none
     * @param hybridRepeat how often each ground-truth box is repeated for them
     * @param anchorQueries whether queries are refined anchor boxes
     * @param auxiliaryLosses whether every layer is supervised
     * @param querySelection whether the decoder queries are the top-scoring encoder tokens
     */
    public record Options(int hybridQueries, int hybridRepeat, boolean anchorQueries, boolean auxiliaryLosses,
            boolean querySelection) {

        /**
         * Checks the options.
         *
         * @param hybridQueries one-to-many queries
         * @param hybridRepeat repeats
         * @param anchorQueries anchor queries
         * @param auxiliaryLosses per-layer losses
         * @param querySelection two-stage query selection
         */
        public Options {
            if (hybridQueries < 0 || hybridQueries > 0 && hybridRepeat < 1) {
                throw new IllegalArgumentException("bad hybrid matching options");
            }
        }

        /**
         * The wave 9 options, without query selection.
         *
         * @param hybridQueries one-to-many queries
         * @param hybridRepeat repeats
         * @param anchorQueries anchor queries
         * @param auxiliaryLosses per-layer losses
         */
        public Options(int hybridQueries, int hybridRepeat, boolean anchorQueries, boolean auxiliaryLosses) {
            this(hybridQueries, hybridRepeat, anchorQueries, auxiliaryLosses, false);
        }

        /** @return everything off */
        public static Options none() {
            return new Options(0, 0, false, false, false);
        }

        /**
         * These options with two-stage query selection switched on or off.
         *
         * @param on whether to select queries from the encoder
         * @return the options
         */
        public Options withQuerySelection(boolean on) {
            return new Options(hybridQueries, hybridRepeat, anchorQueries, auxiliaryLosses, on);
        }

        /** @return whether any option is on */
        public boolean active() {
            return hybridQueries > 0 || anchorQueries || auxiliaryLosses || querySelection;
        }
    }

    private static final int SINE_FREQUENCIES = 4;

    private final int side;
    private final int classes;
    private final int queries;
    private final int dim;
    private final int heads;
    private final boolean denoising;
    private final int maxObjects;
    private final ConvBackbone backbone;
    private final Var project;
    private final Var projectBias;
    private final Var positions;
    private final Var queryEmbed;
    private final Var dnBox;
    private final Var dnLabel;
    private final List<Var[]> layers = new ArrayList<>();
    private final Var clsW;
    private final Var clsB;
    private final Var box1;
    private final Var box1b;
    private final Var box2;
    private final Var box2b;
    private final SplittableRandom noise;
    private final Options options;
    private final Var hybridEmbed;
    private final Var anchorEmbed;
    private final Var posW1;
    private final Var posB1;
    private final Var posW2;
    private final Var posB2;
    private final Var encCls;
    private final Var encClsB;
    private final Var encBox1;
    private final Var encBox1b;
    private final Var encBox2;
    private final Var encBox2b;
    private final Var encNormGain;
    private final Var encNormBias;

    /**
     * Creates the detector.
     *
     * @param side the image side, divisible by 8
     * @param classes the object classes
     * @param queries the object queries, at least the most objects per image
     * @param dim the model width
     * @param layers the decoder layers
     * @param denoising whether to train with denoising queries
     * @param maxObjects the most objects per image (denoising query slots)
     * @param random the source of initial weights and of the denoising noise
     */
    public SetPredictionDetector(int side, int classes, int queries, int dim, int layers, boolean denoising,
            int maxObjects, SplittableRandom random) {
        this(side, classes, queries, dim, layers, denoising, maxObjects, Options.none(), random);
    }

    /**
     * Creates the detector with convergence options. With {@link Options#none()} it is
     * exactly the detector of the other constructor (same weights for the same random
     * source); the options' extra weights are drawn afterwards.
     *
     * @param side the image side, divisible by 8
     * @param classes the object classes
     * @param queries the object queries, at least the most objects per image
     * @param dim the model width
     * @param layers the decoder layers
     * @param denoising whether to train with denoising queries
     * @param maxObjects the most objects per image (denoising query slots)
     * @param options hybrid matching, anchor queries, auxiliary losses
     * @param random the source of initial weights and of the denoising noise
     */
    public SetPredictionDetector(int side, int classes, int queries, int dim, int layers, boolean denoising,
            int maxObjects, Options options, SplittableRandom random) {
        if (denoising && options.active()) {
            throw new IllegalArgumentException("convergence options cannot be combined with denoising queries");
        }
        this.side = side;
        this.classes = classes;
        this.queries = queries;
        this.dim = dim;
        this.heads = 4;
        this.denoising = denoising;
        this.maxObjects = maxObjects;
        this.backbone = new ConvBackbone(1, new int[] {8, 16, 24}, random);
        this.project = VisionOps.convWeight(random, dim, 24, 1);
        this.projectBias = VisionOps.zeros(dim);
        int tokens = (side / 4) * (side / 4);
        this.positions = Var.param(Tensor.normal(Shape.of(1, tokens, dim), 0, 0.1f, random));
        this.queryEmbed = Var.param(Tensor.normal(Shape.of(1, queries, dim), 0, 1f, random));
        this.dnBox = matrix(random, 4, dim);
        this.dnLabel = Var.param(Tensor.normal(Shape.of(classes + 1, dim), 0, 1f, random));
        for (int l = 0; l < layers; l++) {
            this.layers.add(new Var[] {
                matrix(random, dim, 3 * dim), matrix(random, dim, dim), // self attention
                matrix(random, dim, dim), matrix(random, dim, 2 * dim), matrix(random, dim, dim), // cross attention
                matrix(random, dim, 2 * dim), VisionOps.zeros(2 * dim), matrix(random, 2 * dim, dim), VisionOps.zeros(dim),
                ones(dim), VisionOps.zeros(dim), ones(dim), VisionOps.zeros(dim), ones(dim), VisionOps.zeros(dim)});
        }
        this.clsW = matrix(random, dim, classes + 1);
        this.clsB = VisionOps.zeros(classes + 1);
        this.box1 = matrix(random, dim, dim);
        this.box1b = VisionOps.zeros(dim);
        this.box2 = Var.param(Tensor.normal(Shape.of(dim, 4), 0, 0.01f, random));
        this.box2b = VisionOps.zeros(4);
        this.noise = random.split();
        this.options = options;
        this.hybridEmbed = options.hybridQueries() > 0
                ? Var.param(Tensor.normal(Shape.of(1, options.hybridQueries(), dim), 0, 1f, random)) : null;
        if (options.anchorQueries() || options.querySelection()) {
            int total = queries + options.hybridQueries();
            this.anchorEmbed = options.anchorQueries()
                    ? Var.param(Tensor.normal(Shape.of(1, total, 4), 0, 1f, random)) : null;
            this.posW1 = matrix(random, 8 * SINE_FREQUENCIES, dim);
            this.posB1 = VisionOps.zeros(dim);
            this.posW2 = matrix(random, dim, dim);
            this.posB2 = VisionOps.zeros(dim);
        } else {
            this.anchorEmbed = null;
            this.posW1 = null;
            this.posB1 = null;
            this.posW2 = null;
            this.posB2 = null;
        }
        if (options.querySelection()) {
            this.encCls = matrix(random, dim, classes + 1);
            this.encClsB = VisionOps.zeros(classes + 1);
            this.encBox1 = matrix(random, dim, dim);
            this.encBox1b = VisionOps.zeros(dim);
            this.encBox2 = Var.param(Tensor.normal(Shape.of(dim, 4), 0, 0.01f, random));
            this.encBox2b = VisionOps.zeros(4);
            this.encNormGain = ones(dim);
            this.encNormBias = VisionOps.zeros(dim);
        } else {
            this.encCls = null;
            this.encClsB = null;
            this.encBox1 = null;
            this.encBox1b = null;
            this.encBox2 = null;
            this.encBox2b = null;
            this.encNormGain = null;
            this.encNormBias = null;
        }
    }

    /** The reference boxes of the memory tokens: each cell's centre, a fixed size, in logit space. */
    private Tensor tokenReferences(int b, int gh, int gw) {
        float[] r = new float[b * gh * gw * 4];
        for (int i = 0; i < b; i++) {
            for (int y = 0; y < gh; y++) {
                for (int x = 0; x < gw; x++) {
                    int o = ((i * gh + y) * gw + x) * 4;
                    r[o] = logit((x + 0.5f) / gw);
                    r[o + 1] = logit((y + 0.5f) / gh);
                    r[o + 2] = logit(0.2f);
                    r[o + 3] = logit(0.2f);
                }
            }
        }
        return Tensor.of(Shape.of(b, gh * gw, 4), r);
    }

    private static float logit(float p) {
        return (float) Math.log(p / (1 - p));
    }

    /** Encoder predictions of every memory token: {@code [b, tokens, classes + 5]}. */
    private Var encoderOutput(Var memory) {
        int b = memory.shape().dim(0);
        int g = side / 4;
        Var h = norm(memory, encNormGain, encNormBias);
        Var logits = h.matmul(encCls).add(encClsB);
        Var box = h.matmul(encBox1).add(encBox1b).relu().matmul(encBox2).add(encBox2b)
                .add(tokenReferences(b, g, g)).sigmoid();
        return Var.concat(2, logits, box);
    }

    /**
     * Two-stage query selection: the {@code k} memory tokens with the best real-class score.
     *
     * @return {contents [b, k, dim], anchors [b, k, 4] (detached)}
     */
    private Var[] selectQueries(Var memory, Var encoder, int k) {
        int b = memory.shape().dim(0);
        int t = memory.shape().dim(1);
        float[] o = encoder.value().toArray();
        int stride = classes + 5;
        float[] idx = new float[b * k];
        for (int i = 0; i < b; i++) {
            double[] score = new double[t];
            Integer[] order = new Integer[t];
            for (int q = 0; q < t; q++) {
                double[] p = softmax(o, (i * t + q) * stride, classes + 1);
                double best = 0;
                for (int c = 0; c < classes; c++) {
                    best = Math.max(best, p[c]);
                }
                score[q] = best;
                order[q] = q;
            }
            java.util.Arrays.sort(order, (x, y) -> Double.compare(score[y], score[x]));
            for (int j = 0; j < k; j++) {
                idx[i * k + j] = i * t + order[j];
            }
        }
        Tensor index = Tensor.of(Shape.of(b * k), idx);
        Var contents = memory.reshape(b * t, dim).gather(0, index).reshape(b, k, dim);
        Var anchors = encoder.reshape(b * t, stride).slice(1, classes + 1, classes + 5).detach().gather(0, index)
                .reshape(b, k, 4);
        return new Var[] {contents, anchors};
    }

    private static Var matrix(RandomGenerator r, int in, int out) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(1.0 / in), r));
    }

    private static Var ones(int n) {
        return Var.param(Tensor.ones(Shape.of(n)));
    }

    private static Var norm(Var x, Var gain, Var bias) {
        int last = x.shape().rank() - 1;
        Var c = x.sub(x.mean(last, true));
        return c.div(c.square().mean(last, true).add(1e-5f).sqrt()).mul(gain).add(bias);
    }

    private Var attend(Var q, Var k, Var v, Tensor bias) {
        int b = q.shape().dim(0);
        int tq = q.shape().dim(1);
        int tk = k.shape().dim(1);
        int dh = dim / heads;
        Var qh = q.reshape(b, tq, heads, dh).permute(0, 2, 1, 3);
        Var kh = k.reshape(b, tk, heads, dh).permute(0, 2, 1, 3);
        Var vh = v.reshape(b, tk, heads, dh).permute(0, 2, 1, 3);
        return Functional.attention(qh, kh, vh, bias, (float) (1 / Math.sqrt(dh)), 0f, null).permute(0, 2, 1, 3)
                .reshape(b, tq, dim);
    }

    private Var memory(Var images) {
        Var f = backbone.features(images).get(1);
        Var m = VisionOps.conv(f, project, projectBias);
        int b = m.shape().dim(0);
        int t = m.shape().dim(2) * m.shape().dim(3);
        return m.reshape(b, dim, t).permute(0, 2, 1).add(positions);
    }

    /** Runs the decoder on query contents {@code [b, q, dim]} with an optional self-attention mask. */
    private Var decode(Var memory, Var x, Tensor mask) {
        Var mem = memory;
        for (Var[] p : layers) {
            Var h = norm(x, p[9], p[10]);
            Var qkv = h.matmul(p[0]);
            x = x.add(attend(qkv.slice(2, 0, dim), qkv.slice(2, dim, 2 * dim), qkv.slice(2, 2 * dim, 3 * dim), mask)
                    .matmul(p[1]));
            h = norm(x, p[11], p[12]);
            Var kv = mem.matmul(p[3]);
            x = x.add(attend(h.matmul(p[2]), kv.slice(2, 0, dim), kv.slice(2, dim, 2 * dim), null).matmul(p[4]));
            h = norm(x, p[13], p[14]);
            x = x.add(h.matmul(p[5]).add(p[6]).relu().matmul(p[7]).add(p[8]));
        }
        Var logits = x.matmul(clsW).add(clsB);
        Var box = x.matmul(box1).add(box1b).relu().matmul(box2).add(box2b).sigmoid();
        return Var.concat(2, logits, box);
    }

    /**
     * Runs the detector.
     *
     * @param images {@code [batch, 1, side, side]}
     * @return {@code [batch, queries, classes + 1 + 4]}: logits (last is "no object"), then
     *     the box as normalised centre x, centre y, width, height
     */
    @Override
    public Var forward(Var images) {
        int b = images.shape().dim(0);
        Var x = queryEmbed.expand(Shape.of(b, queries, dim));
        if (options.querySelection()) {
            Var mem = memory(images);
            Var[] sel = selectQueries(mem, encoderOutput(mem), queries);
            List<Var> outs = decodeLayers(mem, sel[0], null, sel[1]);
            return outs.get(outs.size() - 1);
        }
        if (options.anchorQueries()) {
            Var anchors = anchorEmbed.slice(1, 0, queries).sigmoid().expand(Shape.of(b, queries, 4));
            List<Var> outs = decodeLayers(memory(images), x, null, anchors);
            return outs.get(outs.size() - 1);
        }
        return decode(memory(images), x, null);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(queries, classes + 5);
    }

    /**
     * The training loss of a batch: Hungarian-matched class and box losses, plus the
     * denoising reconstruction loss when enabled.
     *
     * @param images {@code [batch, 1, side, side]}
     * @param boxes per image the boxes, {@code 4 k} values in pixels
     * @param labels per image the classes
     * @return the loss
     */
    public Var loss(Tensor images, float[][] boxes, int[][] labels) {
        if (options.active()) {
            return optionLoss(images, boxes, labels);
        }
        int b = images.shape().dim(0);
        Var mem = memory(Var.of(images));
        Var x = queryEmbed.expand(Shape.of(b, queries, dim));
        int dn = denoising ? maxObjects : 0;
        Tensor mask = null;
        float[] dnTargets = null;
        float[] dnValid = null;
        float[] dnLabels = null;
        if (dn > 0) {
            float[] noisyBox = new float[b * dn * 4];
            float[] noisyLabel = new float[b * dn];
            dnTargets = new float[b * dn * 4];
            dnValid = new float[b * dn];
            dnLabels = new float[b * dn];
            int total = queries + dn;
            float[] bias = new float[b * total * total];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < dn; j++) {
                    int s = i * dn + j;
                    boolean real = j < labels[i].length;
                    float[] c = real ? centres(boxes[i], j) : new float[] {0.5f, 0.5f, 0.1f, 0.1f};
                    System.arraycopy(c, 0, dnTargets, 4 * s, 4);
                    dnValid[s] = real ? 1 : 0;
                    dnLabels[s] = real ? labels[i][j] : classes;
                    noisyBox[4 * s] = clamp(c[0] + (float) (noise.nextDouble() * 2 - 1) * 0.4f * c[2] / 2);
                    noisyBox[4 * s + 1] = clamp(c[1] + (float) (noise.nextDouble() * 2 - 1) * 0.4f * c[3] / 2);
                    noisyBox[4 * s + 2] = clamp(c[2] * (1 + (float) (noise.nextDouble() * 2 - 1) * 0.4f));
                    noisyBox[4 * s + 3] = clamp(c[3] * (1 + (float) (noise.nextDouble() * 2 - 1) * 0.4f));
                    noisyLabel[s] = !real ? classes : noise.nextDouble() < 0.2 ? noise.nextInt(classes) : labels[i][j];
                }
                for (int q = 0; q < total; q++) {
                    for (int k = 0; k < total; k++) {
                        boolean blocked = q < queries && k >= queries
                                || k >= queries && k - queries >= labels[i].length;
                        bias[(i * total + q) * total + k] = blocked ? -1e9f : 0f;
                    }
                }
            }
            Var dnQueries = Var.of(Tensor.of(Shape.of(b, dn, 4), noisyBox)).matmul(dnBox)
                    .add(dnLabel.gather(0, Tensor.of(Shape.of(b * dn), noisyLabel)).reshape(b, dn, dim));
            x = Var.concat(1, x, dnQueries);
            mask = Tensor.of(Shape.of(b, 1, total, total), bias);
        }
        Var out = decode(mem, x, mask);
        Var match = dn > 0 ? out.slice(1, 0, queries) : out;
        float[] o = match.value().toArray();
        int stride = classes + 5;
        float[] clsTarget = new float[b * queries];
        float[] clsWeight = new float[b * queries];
        List<Integer> matched = new ArrayList<>();
        List<float[]> matchedBoxes = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            int g = labels[i].length;
            double[][] cost = new double[queries][queries];
            for (int q = 0; q < queries; q++) {
                int base = (i * queries + q) * stride;
                double[] p = softmax(o, base, classes + 1);
                float[] pb = {o[base + classes + 1], o[base + classes + 2], o[base + classes + 3], o[base + classes + 4]};
                for (int j = 0; j < g; j++) {
                    float[] t = centres(boxes[i], j);
                    double l1 = 0;
                    for (int d = 0; d < 4; d++) {
                        l1 += Math.abs(pb[d] - t[d]);
                    }
                    cost[q][j] = -p[labels[i][j]] + 5 * l1 + 2 * (1 - giou(corners(pb), corners(t)));
                }
            }
            int[] assign = MinibatchOptimalTransport.assignCost(cost);
            for (int q = 0; q < queries; q++) {
                int s = i * queries + q;
                if (assign[q] < g) {
                    clsTarget[s] = labels[i][assign[q]];
                    clsWeight[s] = 1;
                    matched.add(s);
                    matchedBoxes.add(centres(boxes[i], assign[q]));
                } else {
                    clsTarget[s] = classes;
                    clsWeight[s] = 0.1f;
                }
            }
        }
        Var flat = match.reshape(b * queries, stride);
        Var loss = weightedCe(flat.slice(1, 0, classes + 1), clsTarget, clsWeight);
        if (!matched.isEmpty()) {
            loss = loss.add(boxLoss(flat, matched, matchedBoxes));
        }
        if (dn > 0) {
            Var dnOut = out.slice(1, queries, queries + dn).reshape(b * dn, stride);
            List<Integer> real = new ArrayList<>();
            List<float[]> realBoxes = new ArrayList<>();
            for (int s = 0; s < b * dn; s++) {
                if (dnValid[s] > 0) {
                    real.add(s);
                    realBoxes.add(new float[] {dnTargets[4 * s], dnTargets[4 * s + 1], dnTargets[4 * s + 2],
                        dnTargets[4 * s + 3]});
                }
            }
            if (!real.isEmpty()) {
                loss = loss.add(weightedCe(dnOut.slice(1, 0, classes + 1), dnLabels, dnValid))
                        .add(boxLoss(dnOut, real, realBoxes));
            }
        }
        return loss;
    }

    /** The loss with convergence options: per-layer outputs, hybrid branch, anchor queries. */
    private Var optionLoss(Tensor images, float[][] boxes, int[][] labels) {
        int b = images.shape().dim(0);
        Var mem = memory(Var.of(images));
        int hq = options.hybridQueries();
        int total = queries + hq;
        Var x = queryEmbed.expand(Shape.of(b, queries, dim));
        Tensor mask = null;
        if (hq > 0) {
            x = Var.concat(1, x, hybridEmbed.expand(Shape.of(b, hq, dim)));
            float[] bias = new float[total * total];
            for (int q = 0; q < total; q++) {
                for (int k = 0; k < total; k++) {
                    bias[q * total + k] = (q < queries) != (k < queries) ? -1e9f : 0f;
                }
            }
            mask = Tensor.of(Shape.of(1, 1, total, total), bias);
        }
        Var anchors = options.anchorQueries() ? anchorEmbed.sigmoid().expand(Shape.of(b, total, 4)) : null;
        Var loss = null;
        if (options.querySelection()) {
            Var enc = encoderOutput(mem);
            loss = matchLoss(enc, boxes, labels, 1);
            Var[] sel = selectQueries(mem, enc, queries);
            x = sel[0];
            anchors = sel[1];
            if (hq > 0) {
                Var[] hyb = selectQueries(mem, enc, hq);
                x = Var.concat(1, x, hyb[0]);
                anchors = Var.concat(1, anchors, hyb[1]);
            }
        }
        List<Var> outs = decodeLayers(mem, x, mask, anchors);
        int from = options.auxiliaryLosses() ? 0 : outs.size() - 1;
        for (int l = from; l < outs.size(); l++) {
            Var out = outs.get(l);
            Var term = matchLoss(hq > 0 ? out.slice(1, 0, queries) : out, boxes, labels, 1);
            if (hq > 0) {
                term = term.add(matchLoss(out.slice(1, queries, total), boxes, labels, options.hybridRepeat()));
            }
            loss = loss == null ? term : loss.add(term);
        }
        return loss;
    }

    /** The decoder with optional anchor-box queries; one output per layer. */
    private List<Var> decodeLayers(Var memory, Var x, Tensor mask, Var anchors) {
        List<Var> outs = new ArrayList<>();
        Var anchor = anchors;
        for (Var[] p : layers) {
            Var pos = anchor == null ? null : positional(anchor.detach());
            Var h = norm(x, p[9], p[10]);
            Var hq = pos == null ? h : h.add(pos);
            Var qk = hq.matmul(p[0]);
            Var v = h.matmul(p[0]);
            x = x.add(attend(qk.slice(2, 0, dim), qk.slice(2, dim, 2 * dim), v.slice(2, 2 * dim, 3 * dim), mask)
                    .matmul(p[1]));
            h = norm(x, p[11], p[12]);
            hq = pos == null ? h : h.add(pos);
            Var kv = memory.matmul(p[3]);
            x = x.add(attend(hq.matmul(p[2]), kv.slice(2, 0, dim), kv.slice(2, dim, 2 * dim), null).matmul(p[4]));
            h = norm(x, p[13], p[14]);
            x = x.add(h.matmul(p[5]).add(p[6]).relu().matmul(p[7]).add(p[8]));
            Var logits = x.matmul(clsW).add(clsB);
            Var raw = x.matmul(box1).add(box1b).relu().matmul(box2).add(box2b);
            Var box;
            if (anchor != null) {
                Var a = anchor.detach().clamp(1e-3f, 1 - 1e-3f);
                box = a.log().sub(a.neg().add(1f).log()).add(raw).sigmoid();
                anchor = box;
            } else {
                box = raw.sigmoid();
            }
            outs.add(Var.concat(2, logits, box));
        }
        return outs;
    }

    /** DAB-DETR's positional query: an MLP of the sine embedding of the anchor box. */
    private Var positional(Var anchor) {
        int b = anchor.shape().dim(0);
        int t = anchor.shape().dim(1);
        float[] freq = new float[SINE_FREQUENCIES];
        for (int f = 0; f < SINE_FREQUENCIES; f++) {
            freq[f] = (float) (Math.PI * Math.pow(2, f));
        }
        Var phase = anchor.reshape(b, t, 4, 1).mul(Tensor.of(Shape.of(SINE_FREQUENCIES), freq));
        Var sine = Var.concat(3, phase.sin(), phase.cos()).reshape(b, t, 8 * SINE_FREQUENCIES);
        return sine.matmul(posW1).add(posB1).relu().matmul(posW2).add(posB2);
    }

    /**
     * Hungarian-matched class and box losses of one query group, with every ground-truth
     * box repeated {@code repeat} times (1 is the usual one-to-one matching).
     */
    private Var matchLoss(Var out, float[][] boxes, int[][] labels, int repeat) {
        int b = out.shape().dim(0);
        int nq = out.shape().dim(1);
        float[] o = out.value().toArray();
        int stride = classes + 5;
        float[] clsTarget = new float[b * nq];
        float[] clsWeight = new float[b * nq];
        List<Integer> matched = new ArrayList<>();
        List<float[]> matchedBoxes = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            int g = labels[i].length;
            int targets = Math.min(g * repeat, nq);
            int n = Math.max(nq, targets);
            double[][] cost = new double[n][n];
            for (int q = 0; q < nq; q++) {
                int base = (i * nq + q) * stride;
                double[] p = softmax(o, base, classes + 1);
                float[] pb = {o[base + classes + 1], o[base + classes + 2], o[base + classes + 3], o[base + classes + 4]};
                for (int j = 0; j < targets; j++) {
                    int gt = j % g;
                    float[] t = centres(boxes[i], gt);
                    double l1 = 0;
                    for (int d = 0; d < 4; d++) {
                        l1 += Math.abs(pb[d] - t[d]);
                    }
                    cost[q][j] = -p[labels[i][gt]] + 5 * l1 + 2 * (1 - giou(corners(pb), corners(t)));
                }
            }
            int[] assign = MinibatchOptimalTransport.assignCost(cost);
            for (int q = 0; q < nq; q++) {
                int s = i * nq + q;
                if (assign[q] < targets) {
                    int gt = assign[q] % g;
                    clsTarget[s] = labels[i][gt];
                    clsWeight[s] = 1;
                    matched.add(s);
                    matchedBoxes.add(centres(boxes[i], gt));
                } else {
                    clsTarget[s] = classes;
                    clsWeight[s] = 0.1f;
                }
            }
        }
        Var flat = out.reshape(b * nq, stride);
        Var loss = weightedCe(flat.slice(1, 0, classes + 1), clsTarget, clsWeight);
        if (!matched.isEmpty()) {
            loss = loss.add(boxLoss(flat, matched, matchedBoxes));
        }
        return loss;
    }

    private Var weightedCe(Var logits, float[] target, float[] weight) {
        int n = target.length;
        Var logp = Activation.LOG_SOFTMAX.forward(logits);
        float[] onehot = new float[n * (classes + 1)];
        double total = 0;
        for (int i = 0; i < n; i++) {
            onehot[i * (classes + 1) + (int) target[i]] = weight[i];
            total += weight[i];
        }
        return logp.mul(Tensor.of(Shape.of(n, classes + 1), onehot)).sum().mul((float) (-1 / Math.max(1e-6, total)));
    }

    private Var boxLoss(Var flat, List<Integer> rows, List<float[]> targets) {
        int p = rows.size();
        float[] idx = new float[p];
        float[] t = new float[4 * p];
        for (int i = 0; i < p; i++) {
            idx[i] = rows.get(i);
            System.arraycopy(targets.get(i), 0, t, 4 * i, 4);
        }
        Var pred = flat.slice(1, classes + 1, classes + 5).gather(0, Tensor.of(Shape.of(p), idx));
        Var target = Var.of(Tensor.of(Shape.of(p, 4), t));
        Var l1 = pred.sub(target).abs().mean().mul(4 * 5f);
        return l1.add(Boxes.giouLoss(Boxes.toCorners(pred), Boxes.toCorners(target)).mul(2));
    }

    private float[] centres(float[] boxes, int j) {
        float s = side;
        return new float[] {(boxes[4 * j] + boxes[4 * j + 2]) / 2 / s, (boxes[4 * j + 1] + boxes[4 * j + 3]) / 2 / s,
            (boxes[4 * j + 2] - boxes[4 * j]) / s, (boxes[4 * j + 3] - boxes[4 * j + 1]) / s};
    }

    private static float clamp(float v) {
        return Math.max(0.01f, Math.min(0.99f, v));
    }

    private static float[] corners(float[] c) {
        return new float[] {c[0] - c[2] / 2, c[1] - c[3] / 2, c[0] + c[2] / 2, c[1] + c[3] / 2};
    }

    private static double giou(float[] a, float[] b) {
        double iou = DetectionMetrics.iou(a, b);
        double iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
        double ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        double union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih;
        double hull = (Math.max(a[2], b[2]) - Math.min(a[0], b[0])) * (Math.max(a[3], b[3]) - Math.min(a[1], b[1]));
        return hull <= 0 ? iou : iou - (hull - union) / hull;
    }

    private static double[] softmax(float[] v, int off, int n) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, v[off + i]);
        }
        double[] p = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            p[i] = Math.exp(v[off + i] - max);
            sum += p[i];
        }
        for (int i = 0; i < n; i++) {
            p[i] /= sum;
        }
        return p;
    }

    /**
     * Detects objects: every query whose best real class beats "no object" in score order,
     * with no NMS.
     *
     * @param images {@code [batch, 1, side, side]}
     * @param firstImage the image index of the first image
     * @return one detection per query, scored by its best real class probability
     */
    public List<DetectionMetrics.Detection> detect(Tensor images, int firstImage) {
        int b = images.shape().dim(0);
        float[] o = Var.noGrad(() -> forward(Var.of(images))).value().toArray();
        int stride = classes + 5;
        List<DetectionMetrics.Detection> dets = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            for (int q = 0; q < queries; q++) {
                int base = (i * queries + q) * stride;
                double[] p = softmax(o, base, classes + 1);
                int best = 0;
                for (int c = 1; c < classes; c++) {
                    best = p[c] > p[best] ? c : best;
                }
                float[] c = {o[base + classes + 1], o[base + classes + 2], o[base + classes + 3], o[base + classes + 4]};
                float[] box = corners(c);
                for (int d = 0; d < 4; d++) {
                    box[d] *= side;
                }
                dets.add(new DetectionMetrics.Detection(firstImage + i, best, (float) p[best], box));
            }
        }
        return dets;
    }

    /** @return the multiply-adds of one image (backbone and decoder) */
    public long macs() {
        int t = (side / 4) * (side / 4);
        long m = backbone.macs(side) + VisionOps.convMacs(24, dim, 1, side / 4, side / 4);
        long perLayer = (long) queries * dim * 3 * dim + (long) queries * queries * dim * 2 + (long) queries * dim * dim
                + (long) queries * dim * dim + (long) t * dim * 2 * dim + (long) queries * t * dim * 2
                + (long) queries * dim * dim + (long) queries * dim * 4 * dim;
        long encoder = options.querySelection() ? (long) t * dim * (classes + 1 + dim + 4) : 0;
        return m + layers.size() * perLayer + (long) queries * dim * (classes + 1 + dim + 4) + encoder;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(backbone.parameters());
        p.addAll(List.of(project, projectBias, positions));
        if (!options.querySelection()) {
            p.add(queryEmbed); // selected queries replace the learned constants
        }
        if (denoising) {
            p.add(dnBox);
            p.add(dnLabel);
        }
        for (Var[] l : layers) {
            p.addAll(List.of(l));
        }
        p.addAll(List.of(clsW, clsB, box1, box1b, box2, box2b));
        if (hybridEmbed != null && !options.querySelection()) {
            p.add(hybridEmbed);
        }
        if (anchorEmbed != null) {
            p.add(anchorEmbed);
        }
        if (posW1 != null) {
            p.addAll(List.of(posW1, posB1, posW2, posB2));
        }
        if (encCls != null) {
            p.addAll(List.of(encCls, encClsB, encBox1, encBox1b, encBox2, encBox2b, encNormGain, encNormBias));
        }
        return p;
    }

    @Override
    public String toString() {
        return "SetPredictionDetector(" + queries + " queries, " + layers.size() + " layers"
                + (denoising ? ", denoising" : "") + (options.querySelection() ? ", query selection" : "") + ")";
    }
}
