package io.github.zzuegg.brainiac.experimental.tsc;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Deep convolutional series classifiers as one {@link Layer}: input {@code [batch, length]}
 * (univariate series), output class logits {@code [batch, classes]}.
 * <ul>
 *   <li>{@link Kind#FCN}: the fully convolutional network of Wang, Yan and Oates (IJCNN 2017,
 *       <a href="https://arxiv.org/abs/1611.06455">arXiv:1611.06455</a>): three convolutions
 *       (kernels 8, 5, 3), global average pooling, a linear head;</li>
 *   <li>{@link Kind#RESNET1D}: the same paper's residual network, blocks of three convolutions
 *       with a 1x1 shortcut;</li>
 *   <li>{@link Kind#INCEPTION_TIME}: InceptionTime (Ismail Fawaz et al., DMKD 2020,
 *       <a href="https://arxiv.org/abs/1909.04939">arXiv:1909.04939</a>): inception modules (a 1x1
 *       bottleneck, parallel convolutions of kernel lengths 10, 20 and 40, and a max-pool branch
 *       with a 1x1 convolution) with a residual connection every three modules, and an ensemble of
 *       independently initialised networks whose logits are averaged;</li>
 *   <li>{@link Kind#LITE}: LITE (Ismail-Fawaz et al., DSAA 2023): fixed hand-crafted filters
 *       (increase, decrease and peak detectors of several lengths) beside a multiplexed learned
 *       convolution, then dilated depthwise-separable convolutions, which cuts the parameters by an
 *       order of magnitude.</li>
 * </ul>
 * Widths are scaled by {@code width} (the papers use 128 for FCN, 64 for ResNet, 32 for
 * InceptionTime and LITE) so the networks train in seconds on a CPU. Convolutions use "same"
 * padding; there is no batch normalisation, so outputs depend only on the input series.
 */
public final class DeepSeriesClassifiers implements Layer {

    /** The architecture. */
    public enum Kind {
        /** Fully convolutional network. */
        FCN,
        /** 1-D residual network. */
        RESNET1D,
        /** InceptionTime (an ensemble). */
        INCEPTION_TIME,
        /** LITE with hand-crafted filters and depthwise-separable convolutions. */
        LITE
    }

    /** A convolution's weight and bias with its dilation and groups. */
    private record Conv(Var weight, Var bias, int dilation, int groups) {
    }

    private final Kind kind;
    private final int width;
    private final int classes;
    private final int ensemble;
    private final List<List<Conv>> convs = new ArrayList<>();
    private final List<Var> heads = new ArrayList<>();
    private final List<Var> headBias = new ArrayList<>();
    private final Tensor fixedFilters;
    private final List<Var> parameters = new ArrayList<>();

    /**
     * Creates a network.
     *
     * @param kind the architecture
     * @param width the base number of filters, such as 8 to 32
     * @param classes the number of classes
     * @param ensemble the number of networks averaged (InceptionTime uses 5; 1 for the others)
     * @param seed the initialisation seed
     */
    public DeepSeriesClassifiers(Kind kind, int width, int classes, int ensemble, long seed) {
        this(kind, width, classes, ensemble, 40, seed);
    }

    /**
     * Creates a network with a chosen longest kernel (InceptionTime and LITE use kernels of
     * {@code maxKernel}, half and a quarter of it; 40 in the papers, shorter for short series).
     *
     * @param kind the architecture
     * @param width the base number of filters
     * @param classes the number of classes
     * @param ensemble the number of networks averaged
     * @param maxKernel the longest kernel, at least 8
     * @param seed the initialisation seed
     */
    public DeepSeriesClassifiers(Kind kind, int width, int classes, int ensemble, int maxKernel, long seed) {
        this.kind = kind;
        this.width = width;
        this.classes = classes;
        this.ensemble = Math.max(1, ensemble);
        SplittableRandom r = new SplittableRandom(seed);
        this.fixedFilters = kind == Kind.LITE ? handCrafted() : null;
        int features = 0;
        for (int e = 0; e < this.ensemble; e++) {
            List<Conv> c = new ArrayList<>();
            switch (kind) {
                case FCN -> {
                    c.add(conv(width, 1, 8, 1, 1, r));
                    c.add(conv(2 * width, width, 5, 1, 1, r));
                    c.add(conv(width, 2 * width, 3, 1, 1, r));
                    features = width;
                }
                case RESNET1D -> {
                    int in = 1;
                    for (int b = 0; b < 2; b++) {
                        int out = width * (b + 1);
                        c.add(conv(out, in, 8, 1, 1, r));
                        c.add(conv(out, out, 5, 1, 1, r));
                        c.add(conv(out, out, 3, 1, 1, r));
                        c.add(conv(out, in, 1, 1, 1, r));
                        in = out;
                    }
                    features = in;
                }
                case INCEPTION_TIME -> {
                    int in = 1;
                    for (int m = 0; m < 3; m++) {
                        if (in > 1) {
                            c.add(conv(width, in, 1, 1, 1, r));
                        }
                        int bin = in > 1 ? width : 1;
                        for (int k : new int[] {maxKernel / 4, maxKernel / 2, maxKernel}) {
                            c.add(conv(width, bin, k, 1, 1, r));
                        }
                        c.add(conv(width, in, 1, 1, 1, r));
                        in = 4 * width;
                    }
                    c.add(conv(4 * width, 1, 1, 1, 1, r));
                    features = 4 * width;
                }
                case LITE -> {
                    int hand = fixedFilters.shape().dim(0);
                    for (int k : new int[] {maxKernel, maxKernel / 2, maxKernel / 4}) {
                        c.add(conv(width, 1, k, 1, 1, r));
                    }
                    int in = 3 * width + hand;
                    c.add(conv(in, 1, maxKernel / 2, 2, in, r));
                    c.add(conv(width, in, 1, 1, 1, r));
                    c.add(conv(width, 1, maxKernel / 2, 4, width, r));
                    c.add(conv(width, width, 1, 1, 1, r));
                    features = width;
                }
                default -> throw new IllegalArgumentException(kind.toString());
            }
            convs.add(c);
            Var head = Var.param(Tensor.normal(Shape.of(features, classes), 0, (float) Math.sqrt(1.0 / features), r));
            Var hb = Var.param(Tensor.zeros(Shape.of(classes)));
            heads.add(head);
            headBias.add(hb);
            for (Conv cv : c) {
                parameters.add(cv.weight());
                parameters.add(cv.bias());
            }
            parameters.add(head);
            parameters.add(hb);
        }
    }

    /**
     * A spec for model builders.
     *
     * @param kind the architecture
     * @param width the base number of filters
     * @param classes the number of classes
     * @param ensemble ensemble size
     * @return the spec
     */
    public static LayerSpec spec(Kind kind, int width, int classes, int ensemble) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(classes);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new DeepSeriesClassifiers(kind, width, classes, ensemble, random.nextLong());
            }
        };
    }

    private static Conv conv(int out, int inPerGroup, int k, int dilation, int groups, SplittableRandom r) {
        float std = (float) Math.sqrt(2.0 / (inPerGroup * k));
        return new Conv(Var.param(Tensor.normal(Shape.of(out, inPerGroup, k), 0, std, r)),
                Var.param(Tensor.zeros(Shape.of(out))), dilation, groups);
    }

    /** LITE's fixed filters: increase, decrease and peak detectors of lengths 2 to 32. */
    private static Tensor handCrafted() {
        int maxK = 32;
        List<float[]> filters = new ArrayList<>();
        for (int len = 2; len <= maxK; len *= 2) {
            float[] inc = new float[maxK];
            float[] dec = new float[maxK];
            int off = (maxK - len) / 2;
            for (int t = 0; t < len; t++) {
                float v = t < len / 2 ? -1f : 1f;
                inc[off + t] = v / len;
                dec[off + t] = -v / len;
            }
            filters.add(inc);
            filters.add(dec);
            if (len >= 4) {
                float[] peak = new float[maxK];
                double mean = 0;
                double[] p = new double[len];
                for (int t = 0; t < len; t++) {
                    double u = (t - (len - 1) / 2.0) / (len / 2.0);
                    p[t] = 1 - u * u;
                    mean += p[t] / len;
                }
                for (int t = 0; t < len; t++) {
                    peak[off + t] = (float) ((p[t] - mean) / len);
                }
                filters.add(peak);
            }
        }
        float[] all = new float[filters.size() * maxK];
        for (int i = 0; i < filters.size(); i++) {
            System.arraycopy(filters.get(i), 0, all, i * maxK, maxK);
        }
        return Tensor.of(Shape.of(filters.size(), 1, maxK), all);
    }

    private static Var apply(Var x, Var w, Var b, int dilation, int groups) {
        int k = w.shape().dim(2);
        int pad = dilation * (k - 1);
        return Functional.conv(x, w, b, new int[] {1}, new int[] {pad / 2}, new int[] {pad - pad / 2},
                new int[] {dilation}, groups);
    }

    private static Var apply(Var x, Conv c) {
        return apply(x, c.weight(), c.bias(), c.dilation(), c.groups());
    }

    /** Max pooling with window 3, stride 1 and "same" padding. */
    private static Var maxPool3(Var x) {
        int b = x.shape().dim(0);
        int c = x.shape().dim(1);
        int l = x.shape().dim(2);
        Var pad = Var.of(Tensor.full(Shape.of(b, c, 1), -1e9f));
        Var left = Var.concat(2, pad, x.slice(2, 0, l - 1));
        Var right = Var.concat(2, x.slice(2, 1, l), pad);
        return x.maximum(left).maximum(right);
    }

    /**
     * The logits of each ensemble member.
     *
     * @param input series {@code [batch, length]}
     * @return one {@code [batch, classes]} per member
     */
    public List<Var> memberLogits(Var input) {
        int b = input.shape().dim(0);
        int l = input.shape().dim(1);
        Var x = input.reshape(b, 1, l);
        List<Var> out = new ArrayList<>();
        for (int e = 0; e < ensemble; e++) {
            List<Conv> c = convs.get(e);
            Var h = switch (kind) {
                case FCN -> apply(apply(apply(x, c.get(0)).relu(), c.get(1)).relu(), c.get(2)).relu();
                case RESNET1D -> {
                    Var z = x;
                    for (int blk = 0; blk < 2; blk++) {
                        Var y = apply(z, c.get(4 * blk)).relu();
                        y = apply(y, c.get(4 * blk + 1)).relu();
                        y = apply(y, c.get(4 * blk + 2));
                        z = y.add(apply(z, c.get(4 * blk + 3))).relu();
                    }
                    yield z;
                }
                case INCEPTION_TIME -> {
                    Var z = x;
                    int idx = 0;
                    for (int m = 0; m < 3; m++) {
                        Var in = z;
                        Var bott = m == 0 ? in : apply(in, c.get(idx++));
                        Var a = apply(bott, c.get(idx++));
                        Var bb = apply(bott, c.get(idx++));
                        Var cc = apply(bott, c.get(idx++));
                        Var d = apply(maxPool3(in), c.get(idx++));
                        z = Var.concat(1, a, bb, cc, d).relu();
                    }
                    yield z.add(apply(x, c.get(idx))).relu();
                }
                case LITE -> {
                    Var hand = Functional.conv(x, Var.of(fixedFilters), null, new int[] {1}, new int[] {15},
                            new int[] {16}, new int[] {1}, 1);
                    Var z = Var.concat(1, apply(x, c.get(0)), apply(x, c.get(1)), apply(x, c.get(2)), hand).relu();
                    z = apply(apply(z, c.get(3)), c.get(4)).relu();
                    yield apply(apply(z, c.get(5)), c.get(6)).relu();
                }
            };
            out.add(h.mean(2, false).matmul(heads.get(e)).add(headBias.get(e)));
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        List<Var> m = memberLogits(input);
        Var s = m.get(0);
        for (int e = 1; e < m.size(); e++) {
            s = s.add(m.get(e));
        }
        return m.size() == 1 ? s : s.mul(1f / m.size());
    }

    @Override
    public List<Var> parameters() {
        return List.copyOf(parameters);
    }

    @Override
    public int outputSize(int inputSize) {
        return classes;
    }

    /** @return the number of trainable weights */
    public long parameterCount() {
        long n = 0;
        for (Var p : parameters) {
            n += p.shape().size();
        }
        return n;
    }

    /**
     * Trains with Adam on cross-entropy, each ensemble member on its own loss, minibatches drawn
     * with replacement.
     *
     * @param series training series {@code [n][length]}
     * @param labels their classes
     * @param steps optimisation steps
     * @param batch minibatch size
     * @param learningRate Adam's step
     * @param seed the minibatch seed
     * @return this network
     */
    public DeepSeriesClassifiers fit(float[][] series, int[] labels, int steps, int batch, float learningRate,
            long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        int l = series[0].length;
        int b = Math.min(batch, series.length);
        for (int s = 0; s < steps; s++) {
            float[] xb = new float[b * l];
            float[] yb = new float[b];
            for (int i = 0; i < b; i++) {
                int k = r.nextInt(series.length);
                System.arraycopy(series[k], 0, xb, i * l, l);
                yb[i] = labels[k];
            }
            Tensor ids = Tensor.of(Shape.of(b), yb);
            Var xv = Var.of(Tensor.of(Shape.of(b, l), xb));
            opt.minimize(parameters, () -> {
                Var loss = null;
                for (Var lg : memberLogits(xv)) {
                    Var ce = Functional.crossEntropy(lg, ids);
                    loss = loss == null ? ce : loss.add(ce);
                }
                return loss;
            });
        }
        return this;
    }

    /**
     * Predicts classes.
     *
     * @param series the series {@code [n][length]}
     * @return the classes
     */
    public int[] predict(float[][] series) {
        return WideRidgeClassifier.argmax(scores(series));
    }

    /**
     * Averaged logits.
     *
     * @param series the series {@code [n][length]}
     * @return {@code [n][classes]}
     */
    public double[][] scores(float[][] series) {
        int n = series.length;
        int l = series[0].length;
        float[] all = new float[n * l];
        for (int i = 0; i < n; i++) {
            System.arraycopy(series[i], 0, all, i * l, l);
        }
        float[] z = Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(n, l), all)))).value().toArray();
        double[][] out = new double[n][classes];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < classes; c++) {
                out[i][c] = z[i * classes + c];
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return kind + "(width " + width + ", ensemble " + ensemble + ")";
    }
}
