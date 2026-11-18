package io.github.zzuegg.brainiac.experimental.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A sequential network of the operations microcontroller deployments use, with the introspection
 * edge tools need: convolutions (grouped and depthwise, optional batch norm, optional ReLU),
 * global average pooling and fully connected layers. It is a core {@link Layer} (input
 * {@code [batch, C, H, W]} or {@code [batch, features]}), and exposes each unit's output shape,
 * multiply-adds and parameter count, which {@link MemoryPlanner}, {@link IntegerInference},
 * {@link PatchBasedExecutor} and the training tools read.
 *
 * <p>Quantisation-aware training: {@link #quantAware(boolean)} folds batch norm into the
 * convolutions for good and runs the forward pass with 8-bit per-channel fake-quantised weights
 * and 8-bit fake-quantised activations (ranges tracked by an exponential moving average), with
 * straight-through gradients, so the {@link IntegerInference} conversion afterwards matches
 * (Jacob et al., 2018).
 */
public final class EdgeNet implements Layer {

    /** One operation of the network. */
    public abstract static sealed class Unit permits Conv, Linear, Pool {

        float emaMin;
        float emaMax;
        boolean tracked;

        Unit() {
        }

        /** @return the unit's parameters */
        public abstract List<Var> parameters();
    }

    /** A 2-D convolution, optionally followed by batch norm and ReLU. */
    public static final class Conv extends Unit {
        final int in;
        final int out;
        final int kernel;
        final int stride;
        final int pad;
        final int groups;
        boolean bn;
        final boolean relu;
        final Var weight;
        final Var bias;
        final Var gamma;
        final Var beta;
        final float[] runMean;
        final float[] runVar;

        Conv(int in, int out, int kernel, int stride, int pad, int groups, boolean bn, boolean relu,
                SplittableRandom r) {
            this.in = in;
            this.out = out;
            this.kernel = kernel;
            this.stride = stride;
            this.pad = pad;
            this.groups = groups;
            this.bn = bn;
            this.relu = relu;
            int fanIn = in / groups * kernel * kernel;
            this.weight = Var.param(Tensor.normal(Shape.of(out, in / groups, kernel, kernel), 0,
                    (float) Math.sqrt(2.0 / fanIn), r));
            this.bias = Var.param(Tensor.zeros(Shape.of(out)));
            this.gamma = Var.param(Tensor.ones(Shape.of(out)));
            this.beta = Var.param(Tensor.zeros(Shape.of(out)));
            this.runMean = new float[out];
            this.runVar = new float[out];
            Arrays.fill(runVar, 1);
        }

        /** @return the filters {@code [out, in / groups, k, k]} */
        public Var weight() {
            return weight;
        }

        /** @return the bias */
        public Var bias() {
            return bias;
        }

        /** @return input channels */
        public int inChannels() {
            return in;
        }

        /** @return output channels */
        public int outChannels() {
            return out;
        }

        /** @return the kernel size */
        public int kernel() {
            return kernel;
        }

        /** @return the stride */
        public int stride() {
            return stride;
        }

        /** @return the zero padding on each side */
        public int pad() {
            return pad;
        }

        /** @return the channel groups */
        public int groups() {
            return groups;
        }

        /** @return whether ReLU follows */
        public boolean relu() {
            return relu;
        }

        /** @return whether batch norm follows (false once folded) */
        public boolean batchNorm() {
            return bn;
        }

        @Override
        public List<Var> parameters() {
            return bn ? List.of(weight, bias, gamma, beta) : List.of(weight, bias);
        }

        /**
         * The weights and bias with batch norm (running statistics) folded in.
         *
         * @return {@code {weight, bias}}
         */
        public Tensor[] folded() {
            if (!bn) {
                return new Tensor[] {weight.value(), bias.value()};
            }
            float[] g = gamma.value().toArray();
            float[] be = beta.value().toArray();
            float[] b = bias.value().toArray();
            float[] w = weight.value().toArray();
            int per = w.length / out;
            float[] nw = new float[w.length];
            float[] nb = new float[out];
            for (int c = 0; c < out; c++) {
                float k = (float) (g[c] / Math.sqrt(runVar[c] + 1e-5));
                for (int i = 0; i < per; i++) {
                    nw[c * per + i] = w[c * per + i] * k;
                }
                nb[c] = (b[c] - runMean[c]) * k + be[c];
            }
            return new Tensor[] {Tensor.of(weight.shape(), nw), Tensor.of(Shape.of(out), nb)};
        }
    }

    /** A fully connected layer, optionally followed by ReLU; flattens its input. */
    public static final class Linear extends Unit {
        final int in;
        final int out;
        final boolean relu;
        final Var weight;
        final Var bias;

        Linear(int in, int out, boolean relu, SplittableRandom r) {
            this.in = in;
            this.out = out;
            this.relu = relu;
            this.weight = Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt((relu ? 2.0 : 1.0) / in), r));
            this.bias = Var.param(Tensor.zeros(Shape.of(out)));
        }

        /** @return the weight {@code [in, out]} */
        public Var weight() {
            return weight;
        }

        /** @return the bias */
        public Var bias() {
            return bias;
        }

        /** @return whether ReLU follows */
        public boolean relu() {
            return relu;
        }

        @Override
        public List<Var> parameters() {
            return List.of(weight, bias);
        }
    }

    /** Global average pooling over the spatial axes. */
    public static final class Pool extends Unit {

        Pool() {
        }

        @Override
        public List<Var> parameters() {
            return List.of();
        }
    }

    /** Builds a network unit by unit, tracking shapes. */
    public static final class Builder {
        private final int[] input;
        private int[] shape;
        private final List<Unit> units = new ArrayList<>();
        private final SplittableRandom random;

        Builder(long seed, int... input) {
            this.input = input.clone();
            this.shape = input.clone();
            this.random = new SplittableRandom(seed);
        }

        /**
         * Adds a convolution with batch norm and ReLU.
         *
         * @param out output channels
         * @param kernel kernel size (odd)
         * @param stride stride
         * @return this builder
         */
        public Builder conv(int out, int kernel, int stride) {
            return conv(out, kernel, stride, 1, true, true);
        }

        /**
         * Adds a convolution with "same"-style padding {@code kernel / 2}.
         *
         * @param out output channels
         * @param kernel kernel size
         * @param stride stride
         * @param groups channel groups ({@code in} for depthwise)
         * @param bn batch norm
         * @param relu ReLU
         * @return this builder
         */
        public Builder conv(int out, int kernel, int stride, int groups, boolean bn, boolean relu) {
            if (shape.length != 3) {
                throw new IllegalStateException("convolution needs an image input");
            }
            int pad = kernel / 2;
            units.add(new Conv(shape[0], out, kernel, stride, pad, groups, bn, relu, random));
            shape = new int[] {out, (shape[1] + 2 * pad - kernel) / stride + 1, (shape[2] + 2 * pad - kernel) / stride + 1};
            return this;
        }

        /**
         * Adds a depthwise-separable block: depthwise {@code k x k} then pointwise {@code 1 x 1},
         * each with batch norm and ReLU.
         *
         * @param out output channels
         * @param kernel depthwise kernel size
         * @param stride depthwise stride
         * @return this builder
         */
        public Builder separable(int out, int kernel, int stride) {
            conv(shape[0], kernel, stride, shape[0], true, true);
            return conv(out, 1, 1, 1, true, true);
        }

        /** @return this builder after global average pooling */
        public Builder pool() {
            units.add(new Pool());
            shape = new int[] {shape[0]};
            return this;
        }

        /**
         * Adds a fully connected layer (flattening an image input).
         *
         * @param out outputs
         * @param relu ReLU
         * @return this builder
         */
        public Builder dense(int out, boolean relu) {
            int in = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
            units.add(new Linear(in, out, relu, random));
            shape = new int[] {out};
            return this;
        }

        /** @return the network */
        public EdgeNet build() {
            return new EdgeNet(input, units);
        }
    }

    private final int[] input;
    private final List<Unit> units;
    private boolean quantAware;

    private final java.util.Map<Unit, java.util.function.Supplier<Var>> deltas = new java.util.IdentityHashMap<>();
    private final List<java.util.function.Function<Var, Var>> branches;

    private EdgeNet(int[] input, List<Unit> units) {
        this.input = input.clone();
        this.units = List.copyOf(units);
        this.branches = new ArrayList<>(java.util.Collections.nCopies(units.size(), null));
    }

    /**
     * Adds a trainable term to a unit's weight (a low-rank adapter, for example), or removes it.
     *
     * @param unit a convolution or fully connected unit
     * @param delta builds a value of the weight's shape at each forward pass, or {@code null} to
     *     remove
     */
    public void setWeightDelta(Unit unit, java.util.function.Supplier<Var> delta) {
        if (delta == null) {
            deltas.remove(unit);
        } else {
            deltas.put(unit, delta);
        }
    }

    /**
     * Adds a side branch to a unit: its output becomes {@code unit(x) + branch(x)} (a lite
     * residual, for example), or removes it.
     *
     * @param index the unit
     * @param branch maps the unit's input to a tensor of the unit's output shape, or {@code null}
     */
    public void setBranch(int index, java.util.function.Function<Var, Var> branch) {
        branches.set(index, branch);
    }

    private Var weightOf(Unit u, Var w) {
        java.util.function.Supplier<Var> d = deltas.get(u);
        return d == null ? w : w.add(d.get());
    }

    /**
     * Starts a network.
     *
     * @param seed the initialisation seed
     * @param input the per-sample input shape: {@code C, H, W} or {@code features}
     * @return a builder
     */
    public static Builder builder(long seed, int... input) {
        return new Builder(seed, input);
    }

    /** @return the per-sample input shape */
    public int[] inputShape() {
        return input.clone();
    }

    /** @return the units in order */
    public List<Unit> units() {
        return units;
    }

    /**
     * The per-sample shape after each unit.
     *
     * @return {@code shapes[i]} after unit {@code i}; {@code shapes[-1]} is not included
     */
    public List<int[]> shapes() {
        List<int[]> out = new ArrayList<>();
        int[] s = input.clone();
        for (Unit u : units) {
            s = switch (u) {
                case Conv c -> new int[] {c.out, (s[1] + 2 * c.pad - c.kernel) / c.stride + 1,
                    (s[2] + 2 * c.pad - c.kernel) / c.stride + 1};
                case Linear l -> new int[] {l.out};
                case Pool p -> new int[] {s[0]};
            };
            out.add(s);
        }
        return out;
    }

    /**
     * Elements of a per-sample shape.
     *
     * @param shape the shape
     * @return its size
     */
    public static long size(int[] shape) {
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }

    /** @return multiply-adds per sample of each unit */
    public long[] macsPerUnit() {
        List<int[]> shapes = shapes();
        long[] m = new long[units.size()];
        for (int i = 0; i < units.size(); i++) {
            int[] o = shapes.get(i);
            m[i] = switch (units.get(i)) {
                case Conv c -> (long) c.out * o[1] * o[2] * (c.in / c.groups) * c.kernel * c.kernel;
                case Linear l -> (long) l.in * l.out;
                case Pool p -> 0;
            };
        }
        return m;
    }

    /** @return multiply-adds per sample */
    public long macs() {
        return Arrays.stream(macsPerUnit()).sum();
    }

    /** @return the number of weights and biases (batch norm folded) */
    public long parameterCount() {
        long n = 0;
        for (Unit u : units) {
            switch (u) {
                case Conv c -> n += c.weight.value().size() + c.out;
                case Linear l -> n += l.weight.value().size() + l.out;
                case Pool p -> {
                }
            }
        }
        return n;
    }

    /**
     * Switches quantisation-aware training on: batch norm is folded into the convolutions (once),
     * and the forward pass fake-quantises weights and activations to 8 bits.
     *
     * @param on whether to fake-quantise
     * @return this network
     */
    public EdgeNet quantAware(boolean on) {
        if (on) {
            foldBatchNorm();
        }
        this.quantAware = on;
        return this;
    }

    /** @return whether the forward pass fake-quantises */
    public boolean isQuantAware() {
        return quantAware;
    }

    /**
     * Folds batch norm (running statistics) into the convolution weights and biases for good.
     *
     * @return this network
     */
    public EdgeNet foldBatchNorm() {
        for (Unit u : units) {
            if (u instanceof Conv c && c.bn) {
                Tensor[] f = c.folded();
                c.weight.assign(f[0]);
                c.bias.assign(f[1]);
                c.bn = false;
            }
        }
        return this;
    }

    /** @return the range tracker of the network input (package use) */
    Unit inputRange() {
        return inputRange;
    }

    @Override
    public Var forward(Var x) {
        return forward(x, false);
    }

    @Override
    public Var forward(Var x, boolean training) {
        return forwardUnits(x, 0, units.size(), training);
    }

    /**
     * Runs units {@code [from, to)}.
     *
     * @param x their input
     * @param from the first unit
     * @param to one past the last unit
     * @param training batch statistics for batch norm, and range tracking under QAT
     * @return the output
     */
    public Var forwardUnits(Var x, int from, int to, boolean training) {
        Var h = x;
        if (quantAware && from == 0) {
            h = fakeQuantActivation(h, inputRange, training);
        }
        for (int i = from; i < to; i++) {
            Unit u = units.get(i);
            Var in = h;
            h = switch (u) {
                case Conv c -> conv(c, h, training);
                case Linear l -> {
                    Var flat = h.reshape(h.shape().dim(0), l.in);
                    Var w = quantAware ? fakeQuantWeight(weightOf(l, l.weight), false) : weightOf(l, l.weight);
                    Var y = flat.matmul(w).add(l.bias);
                    yield l.relu ? y.relu() : y;
                }
                case Pool p -> h.mean(3, false).mean(2, false);
            };
            if (branches.get(i) != null) {
                h = h.add(branches.get(i).apply(in));
            }
            if (quantAware) {
                h = fakeQuantActivation(h, u, training);
            }
        }
        return h;
    }

    private final Unit inputRange = new Pool();

    private Var conv(Conv c, Var x, boolean training) {
        Var w = quantAware ? fakeQuantWeight(weightOf(c, c.weight), true) : weightOf(c, c.weight);
        Var y = Functional.conv(x, w, c.bias, new int[] {c.stride, c.stride}, new int[] {c.pad, c.pad},
                new int[] {c.pad, c.pad}, new int[] {1, 1}, c.groups);
        if (c.bn) {
            int n = c.out;
            if (training) {
                Var mean = y.mean(0, true).mean(2, true).mean(3, true);
                Var centred = y.sub(mean);
                Var var = centred.square().mean(0, true).mean(2, true).mean(3, true);
                y = centred.div(var.add(1e-5f).sqrt());
                float[] m = mean.value().toArray();
                float[] v = var.value().toArray();
                for (int i = 0; i < n; i++) {
                    c.runMean[i] = 0.9f * c.runMean[i] + 0.1f * m[i];
                    c.runVar[i] = 0.9f * c.runVar[i] + 0.1f * v[i];
                }
            } else {
                float[] inv = new float[n];
                for (int i = 0; i < n; i++) {
                    inv[i] = (float) (1 / Math.sqrt(c.runVar[i] + 1e-5));
                }
                y = y.sub(Tensor.of(Shape.of(1, n, 1, 1), c.runMean)).mul(Tensor.of(Shape.of(1, n, 1, 1), inv));
            }
            y = y.mul(c.gamma.reshape(1, n, 1, 1)).add(c.beta.reshape(1, n, 1, 1));
        }
        return c.relu ? y.relu() : y;
    }

    /** Per-output-channel symmetric 8-bit fake quantisation with a straight-through gradient. */
    private static Var fakeQuantWeight(Var w, boolean convLayout) {
        Tensor q = IntegerInference.fakeQuantWeights(w.value(), convLayout);
        return Functional.straightThrough(Var.of(q), w);
    }

    private static Var fakeQuantActivation(Var h, Unit range, boolean training) {
        if (training || !range.tracked) {
            float[] v = h.value().toArray();
            float lo = 0;
            float hi = 0;
            for (float f : v) {
                lo = Math.min(lo, f);
                hi = Math.max(hi, f);
            }
            if (!range.tracked) {
                range.emaMin = lo;
                range.emaMax = hi;
                range.tracked = true;
            } else {
                range.emaMin = 0.95f * range.emaMin + 0.05f * lo;
                range.emaMax = 0.95f * range.emaMax + 0.05f * hi;
            }
        }
        float[] sz = IntegerInference.affineParams(range.emaMin, range.emaMax);
        float s = sz[0];
        float z = sz[1];
        float[] v = h.value().toArray();
        float[] q = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            float qi = Math.max(-128, Math.min(127, Math.round(v[i] / s) + z));
            q[i] = (qi - z) * s;
        }
        return Functional.straightThrough(Var.of(Tensor.of(h.shape(), q)), h);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        units.forEach(u -> p.addAll(u.parameters()));
        return p;
    }

    /**
     * Trains with Adam and cross-entropy on random mini-batches.
     *
     * @param x inputs {@code [n, ...]}
     * @param y class ids {@code [n]}
     * @param steps steps
     * @param batch batch size
     * @param lr the learning rate
     * @param trainable the parameters to update
     * @param seed the seed
     * @return the last loss
     */
    public float train(Tensor x, Tensor y, int steps, int batch, float lr, List<Var> trainable, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.shape().dim(0);
        float last = 0;
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(Math.max(1, n - batch));
            int to = Math.min(n, from + batch);
            for (Var p : parameters()) {
                p.zeroGrad();
            }
            Var loss = Functional.crossEntropy(forward(Var.of(x.slice(0, from, to)), true), y.slice(0, from, to));
            last = loss.value().item();
            loss.backward();
            opt.step(trainable);
            for (Var p : parameters()) {
                p.zeroGrad();
            }
        }
        return last;
    }

    /**
     * Classification accuracy in evaluation mode.
     *
     * @param x inputs
     * @param y class ids
     * @return the share classified correctly
     */
    public double accuracy(Tensor x, Tensor y) {
        float[] z = Var.noGrad(() -> forward(Var.of(x), false)).value().toArray();
        float[] ids = y.toArray();
        int k = z.length / ids.length;
        int ok = 0;
        for (int i = 0; i < ids.length; i++) {
            int best = 0;
            for (int c = 1; c < k; c++) {
                if (z[i * k + c] > z[i * k + best]) {
                    best = c;
                }
            }
            ok += best == (int) ids[i] ? 1 : 0;
        }
        return ok / (double) ids.length;
    }
}
