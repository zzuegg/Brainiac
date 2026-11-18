package io.github.zzuegg.brainiac.experimental.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A once-for-all supernet of MobileNet-style inverted-residual (MBConv) blocks with elastic
 * kernel size, depth and width, trained once by progressive shrinking so that its subnets work
 * without retraining.
 *
 * <p>Structure: a 3x3 stem, then stages of up to {@code maxDepth} blocks; each block is a 1x1
 * expansion, a depthwise {@code k x k} convolution (stride 2 in a stage's first block) and a 1x1
 * projection, with a residual connection when the shapes allow. Elastic choices:
 * <ul>
 *   <li>kernel 5 or 3: the 3x3 kernel is the centre of the 5x5 one passed through a learned
 *       9x9 transformation matrix per block;</li>
 *   <li>depth: a stage keeps its first {@code d} blocks;</li>
 *   <li>width: the expansion ratio; channels are sorted by the L1 norm of their expansion
 *       weights before width shrinking, so a narrower block keeps the most important ones.</li>
 * </ul>
 * Progressive shrinking trains the largest network, then samples elastic kernels, then depths,
 * then widths, with knowledge distillation from the largest network's outputs. Unlike OFA, whose
 * teacher is a fully trained, frozen network, the teacher here is the supernet's own largest
 * path, and during the elastic phases every other step trains that path directly (the
 * "sandwich" rule of later supernet work); without that, short runs collapsed.
 *
 * <p>Paper: Cai et al., "Once-for-All: Train One Network and Specialize it for Efficient
 * Deployment", ICLR 2020, <a href="https://arxiv.org/abs/1908.09791">arXiv:1908.09791</a>.
 */
public final class ElasticSupernet {

    /**
     * A subnet.
     *
     * @param depths blocks per stage
     * @param kernels per block slot ({@code stage * maxDepth + block}), 3 or 5
     * @param expands per block slot, the expansion ratio
     */
    public record Config(int[] depths, int[] kernels, int[] expands) {

        @Override
        public String toString() {
            return "d" + Arrays.toString(depths) + " k" + Arrays.toString(kernels) + " e" + Arrays.toString(expands);
        }
    }

    /** How the supernet is trained. */
    public enum Schedule {
        /** only the largest network */
        LARGEST,
        /** progressive shrinking with distillation (kernel, then depth, then width) */
        PROGRESSIVE,
        /** every elastic dimension sampled uniformly from the start, no distillation */
        RANDOM
    }

    private final int side;
    private final int inChannels;
    private final int classes;
    private final int stem;
    private final int[] widths;
    private final int maxDepth;
    private final int maxExpand;
    private final Var stemW;
    private final Var stemB;
    private final List<Var[]> blocks = new ArrayList<>();   // {expandW, expandB, dwW, dwB, transform, projW, projB}
    private final Var head;
    private final Var headB;

    /**
     * Creates a supernet.
     *
     * @param inChannels input channels
     * @param side input side
     * @param classes classes
     * @param stem stem channels
     * @param widths output channels of each stage
     * @param maxDepth largest blocks per stage
     * @param maxExpand largest expansion ratio
     * @param seed the seed
     */
    public ElasticSupernet(int inChannels, int side, int classes, int stem, int[] widths, int maxDepth, int maxExpand,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.side = side;
        this.inChannels = inChannels;
        this.classes = classes;
        this.stem = stem;
        this.widths = widths.clone();
        this.maxDepth = maxDepth;
        this.maxExpand = maxExpand;
        stemW = Var.param(Tensor.normal(Shape.of(stem, inChannels, 3, 3), 0, (float) Math.sqrt(2.0 / (9 * inChannels)), r));
        stemB = Var.param(Tensor.zeros(Shape.of(stem)));
        int cin = stem;
        for (int s = 0; s < widths.length; s++) {
            for (int b = 0; b < maxDepth; b++) {
                int in = b == 0 ? cin : widths[s];
                int hid = in * maxExpand;
                float[] eye = new float[81];
                for (int i = 0; i < 9; i++) {
                    eye[i * 9 + i] = 1;
                }
                blocks.add(new Var[] {
                    Var.param(Tensor.normal(Shape.of(hid, in, 1, 1), 0, (float) Math.sqrt(2.0 / in), r)),
                    Var.param(Tensor.zeros(Shape.of(hid))),
                    Var.param(Tensor.normal(Shape.of(hid, 1, 5, 5), 0, (float) Math.sqrt(2.0 / 25), r)),
                    Var.param(Tensor.zeros(Shape.of(hid))),
                    Var.param(Tensor.of(Shape.of(9, 9), eye)),
                    Var.param(Tensor.normal(Shape.of(widths[s], hid, 1, 1), 0,
                            (float) Math.sqrt((b == 0 ? 2.0 : 0.5) / hid), r)),
                    Var.param(Tensor.zeros(Shape.of(widths[s])))});
            }
            cin = widths[s];
        }
        head = Var.param(Tensor.normal(Shape.of(widths[widths.length - 1], classes), 0,
                (float) Math.sqrt(1.0 / widths[widths.length - 1]), r));
        headB = Var.param(Tensor.zeros(Shape.of(classes)));
    }

    /** @return all parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(stemW, stemB, head, headB));
        blocks.forEach(b -> p.addAll(List.of(b)));
        return p;
    }

    /** @return the number of block slots */
    public int slots() {
        return widths.length * maxDepth;
    }

    /** @return the stages */
    public int stages() {
        return widths.length;
    }

    /** @return the largest blocks per stage */
    public int maxDepth() {
        return maxDepth;
    }

    /** @return the largest subnet */
    public Config largest() {
        int[] d = new int[widths.length];
        Arrays.fill(d, maxDepth);
        int[] k = new int[slots()];
        Arrays.fill(k, 5);
        int[] e = new int[slots()];
        Arrays.fill(e, maxExpand);
        return new Config(d, k, e);
    }

    /**
     * Samples a subnet.
     *
     * @param r the random source
     * @param kernels allowed kernels
     * @param depths allowed depths
     * @param expands allowed expansion ratios
     * @return the subnet
     */
    public Config sample(SplittableRandom r, int[] kernels, int[] depths, int[] expands) {
        int[] d = new int[widths.length];
        for (int s = 0; s < d.length; s++) {
            d[s] = depths[r.nextInt(depths.length)];
        }
        int[] k = new int[slots()];
        int[] e = new int[slots()];
        for (int i = 0; i < k.length; i++) {
            k[i] = kernels[r.nextInt(kernels.length)];
            e[i] = expands[r.nextInt(expands.length)];
        }
        return new Config(d, k, e);
    }

    /**
     * Forward pass of a subnet.
     *
     * @param x images {@code [n, C, side, side]}
     * @param c the subnet
     * @return logits
     */
    public Var forward(Var x, Config c) {
        Var h = Functional.conv(x, stemW, stemB, new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1},
                new int[] {1, 1}, 1).relu();
        int cin = stem;
        for (int s = 0; s < widths.length; s++) {
            for (int b = 0; b < c.depths()[s]; b++) {
                int slot = s * maxDepth + b;
                Var[] p = blocks.get(slot);
                int hid = cin * c.expands()[slot];
                int stride = b == 0 ? 2 : 1;
                Var e = Functional.conv(h, p[0].slice(0, 0, hid), p[1].slice(0, 0, hid), new int[] {1, 1},
                        new int[] {0, 0}, new int[] {0, 0}, new int[] {1, 1}, 1).relu();
                Var dw = p[2].slice(0, 0, hid);
                int k = c.kernels()[slot];
                if (k == 3) {
                    dw = dw.slice(2, 1, 4).slice(3, 1, 4).reshape(hid, 9).matmul(p[4]).reshape(hid, 1, 3, 3);
                }
                Var d = Functional.conv(e, dw, p[3].slice(0, 0, hid), new int[] {stride, stride},
                        new int[] {k / 2, k / 2}, new int[] {k / 2, k / 2}, new int[] {1, 1}, hid).relu();
                Var y = Functional.conv(d, p[5].slice(1, 0, hid), p[6], new int[] {1, 1}, new int[] {0, 0},
                        new int[] {0, 0}, new int[] {1, 1}, 1);
                h = b == 0 ? y.relu() : h.add(y);
                cin = widths[s];
            }
        }
        return h.mean(3, false).mean(2, false).matmul(head).add(headB);
    }

    /**
     * Sorts each block's hidden channels by the L1 norm of their expansion weights, descending,
     * so that width shrinking keeps the most important channels (the function is unchanged).
     */
    public void sortChannels() {
        for (Var[] p : blocks) {
            float[] ew = p[0].value().toArray();
            int hid = p[0].shape().dim(0);
            int in = p[0].shape().dim(1);
            double[] norm = new double[hid];
            for (int h = 0; h < hid; h++) {
                for (int i = 0; i < in; i++) {
                    norm[h] += Math.abs(ew[h * in + i]);
                }
            }
            Integer[] order = new Integer[hid];
            for (int i = 0; i < hid; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(norm[b], norm[a]));
            permuteRows(p[0], order);
            permuteRows(p[1], order);
            permuteRows(p[2], order);
            permuteRows(p[3], order);
            float[] pw = p[5].value().toArray();
            int out = p[5].shape().dim(0);
            float[] npw = new float[pw.length];
            for (int o = 0; o < out; o++) {
                for (int h = 0; h < hid; h++) {
                    npw[o * hid + h] = pw[o * hid + order[h]];
                }
            }
            p[5].assign(Tensor.of(p[5].shape(), npw));
        }
    }

    private static void permuteRows(Var v, Integer[] order) {
        float[] a = v.value().toArray();
        int rows = v.shape().dim(0);
        int per = a.length / rows;
        float[] b = new float[a.length];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(a, order[r] * per, b, r * per, per);
        }
        v.assign(Tensor.of(v.shape(), b));
    }

    /**
     * Trains the supernet.
     *
     * @param x images
     * @param y class ids
     * @param steps steps in total (progressive shrinking gives half to the largest network and a
     *     sixth to each elastic phase)
     * @param schedule how
     * @param kernels the kernel choices
     * @param depths the depth choices
     * @param expands the expansion choices
     * @param seed the seed
     * @return the last loss
     */
    public float train(Tensor x, Tensor y, int steps, Schedule schedule, int[] kernels, int[] depths, int[] expands,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Config big = largest();
        int[] maxK = {5};
        int[] maxD = {maxDepth};
        int[] maxE = {maxExpand};
        int full = Math.max(1, steps / 2);
        int phase = Math.max(1, (steps - full) / 3);
        Optimizer opt = Optimizer.adam(1e-2f);
        float last = 0;
        boolean sorted = false;
        int lastPhase = 0;
        for (int s = 0; s < steps; s++) {
            Config c;
            boolean kd = false;
            switch (schedule) {
                case LARGEST -> c = big;
                case RANDOM -> c = sample(r, kernels, depths, expands);
                default -> {
                    int ph = s < full ? 0 : Math.min(3, 1 + (s - full) / phase);
                    if (ph == 3 && !sorted) {
                        sortChannels();
                        sorted = true;
                    }
                    if (ph != lastPhase) {
                        // a fresh optimizer per phase: Adam's moments would not follow the channel
                        // permutation of sortChannels (stale moments wrecked the network)
                        opt = Optimizer.adam(1e-2f);
                        lastPhase = ph;
                    }
                    c = switch (ph) {
                        case 0 -> big;
                        case 1 -> sample(r, kernels, maxD, maxE);
                        case 2 -> sample(r, kernels, depths, maxE);
                        default -> sample(r, kernels, depths, expands);
                    };
                    kd = ph > 0;
                    if (ph > 0 && s % 2 == 1) {
                        // every other step trains the largest network itself (no distillation), so the
                        // short elastic phases of a small budget do not wreck the teacher
                        c = big;
                        kd = false;
                    }
                }
            }
            last = step(x, y, c, kd ? big : null, opt, r);
        }
        return last;
    }

    /**
     * Trains one fixed subnet (the from-scratch baseline).
     *
     * @param x images
     * @param y class ids
     * @param steps steps
     * @param c the subnet
     * @param seed the seed
     * @return the last loss
     */
    public float trainFixed(Tensor x, Tensor y, int steps, Config c, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        float last = 0;
        for (int s = 0; s < steps; s++) {
            last = step(x, y, c, null, opt, r);
        }
        return last;
    }

    private float step(Tensor x, Tensor y, Config c, Config teacher, Optimizer opt, SplittableRandom r) {
        int n = x.shape().dim(0);
        int batch = Math.min(16, n);
        int from = r.nextInt(Math.max(1, n - batch));
        Tensor bx = x.slice(0, from, from + batch);
        Tensor by = y.slice(0, from, from + batch);
        List<Var> params = parameters();
        params.forEach(Var::zeroGrad);
        Var logits = forward(Var.of(bx), c);
        Var loss = Functional.crossEntropy(logits, by);
        if (teacher != null) {
            float[] t = Var.noGrad(() -> forward(Var.of(bx), teacher)).value().toArray();
            float[] p = new float[t.length];
            for (int i = 0; i < batch; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < classes; k++) {
                    max = Math.max(max, t[i * classes + k]);
                }
                double z = 0;
                for (int k = 0; k < classes; k++) {
                    z += Math.exp(t[i * classes + k] - max);
                }
                for (int k = 0; k < classes; k++) {
                    p[i * classes + k] = (float) (Math.exp(t[i * classes + k] - max) / z);
                }
            }
            Var soft = Activation.LOG_SOFTMAX.forward(logits).mul(Tensor.of(Shape.of(batch, classes), p)).sum()
                    .mul(-1f / batch);
            loss = loss.add(soft);
        }
        float v = loss.value().item();
        loss.backward();
        opt.step(params);
        params.forEach(Var::zeroGrad);
        return v;
    }

    /**
     * Accuracy of a subnet.
     *
     * @param c the subnet
     * @param x images
     * @param y class ids
     * @return the share correct
     */
    public double accuracy(Config c, Tensor x, Tensor y) {
        float[] z = Var.noGrad(() -> forward(Var.of(x), c)).value().toArray();
        float[] ids = y.toArray();
        int ok = 0;
        for (int i = 0; i < ids.length; i++) {
            int best = 0;
            for (int k = 1; k < classes; k++) {
                if (z[i * classes + k] > z[i * classes + best]) {
                    best = k;
                }
            }
            ok += best == (int) ids[i] ? 1 : 0;
        }
        return ok / (double) ids.length;
    }

    /**
     * Multiply-adds per sample of a subnet.
     *
     * @param c the subnet
     * @return the count
     */
    public long macs(Config c) {
        return cost(c)[0];
    }

    /**
     * Peak int8 activation bytes of a subnet run layer by layer.
     *
     * @param c the subnet
     * @return the peak
     */
    public long peakBytes(Config c) {
        return cost(c)[1];
    }

    /**
     * Parameters of a subnet.
     *
     * @param c the subnet
     * @return the count
     */
    public long parameterCount(Config c) {
        return cost(c)[2];
    }

    private long[] cost(Config c) {
        long hw = (long) side * side;
        long macs = hw * stem * inChannels * 9;
        long params = (long) stem * inChannels * 9 + stem;
        long peak = hw * (inChannels + stem);
        int cin = stem;
        int s0 = side;
        for (int s = 0; s < widths.length; s++) {
            for (int b = 0; b < c.depths()[s]; b++) {
                int slot = s * maxDepth + b;
                int hid = cin * c.expands()[slot];
                int k = c.kernels()[slot];
                int s1 = b == 0 ? (s0 + 1) / 2 : s0;
                long inA = (long) s0 * s0;
                long outA = (long) s1 * s1;
                macs += inA * cin * hid + outA * hid * k * k + outA * hid * widths[s];
                params += (long) cin * hid + hid + (long) hid * k * k + hid + (long) hid * widths[s] + widths[s];
                long keep = b == 0 ? 0 : outA * cin;        // the residual input stays alive
                peak = Math.max(peak, inA * cin + inA * hid);
                peak = Math.max(peak, inA * hid + outA * hid + keep);
                peak = Math.max(peak, outA * hid + outA * widths[s] + keep);
                cin = widths[s];
                s0 = s1;
            }
        }
        macs += (long) cin * classes;
        params += (long) cin * classes + classes;
        return new long[] {macs, peak, params};
    }

    /**
     * The size of the configuration space, for reporting.
     *
     * @param kernels kernel choices
     * @param depths depth choices
     * @param expands expansion choices
     * @return the number of distinct subnets (blocks beyond a stage's depth do not count)
     */
    public double spaceSize(int[] kernels, int[] depths, int[] expands) {
        double perStage = 0;
        for (int d : depths) {
            perStage += Math.pow(kernels.length * expands.length, d);
        }
        return Math.pow(perStage, widths.length);
    }
}
