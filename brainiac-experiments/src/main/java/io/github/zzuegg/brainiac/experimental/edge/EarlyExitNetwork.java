package io.github.zzuegg.brainiac.experimental.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Early-exit networks: a backbone of blocks with an internal classifier ("exit head") after each
 * block; at inference the input stops at the first exit whose confidence passes a threshold, so
 * easy inputs use a fraction of the compute.
 *
 * <ul>
 *   <li>Training: {@link #trainJoint} sums the weighted cross-entropy of every exit, as BranchyNet
 *       (Teerapittayanon, McDanel and Kung, ICPR 2016); {@link #trainFrozen} first trains the
 *       backbone with its final head and then only the internal heads on the frozen features,
 *       as shallow-deep networks do for a pretrained model (Kaya, Hong and Dumitras, ICML 2019,
 *       <a href="https://arxiv.org/abs/1810.07052">arXiv:1810.07052</a>).</li>
 *   <li>Exit rules ({@link Rule}): maximum probability above a threshold, entropy below it, or
 *       patience (the same prediction at {@code k} consecutive exits).</li>
 *   <li>Diagnostics: the overthinking rate (some earlier exit is right but the last is wrong),
 *       the area under the accuracy-compute curve, and EEFP (Kubaty et al., NeurIPS 2025,
 *       <a href="https://arxiv.org/abs/2508.21495">arXiv:2508.21495</a>): per exit, the AUROC of
 *       its confidence against the target "should stop" (the exit is right, or no later exit
 *       is), which judges exits by failure prediction rather than calibration.</li>
 * </ul>
 *
 * <p>Two backbones: {@link #mlp} (dense tanh blocks) and {@link #cnn} (3×3 ReLU convolutions, stride
 * 1 in the first block and 2 after, with global-average-pool heads).
 */
public final class EarlyExitNetwork {

    /** The exit rule. */
    public enum Rule {
        /** stop when the largest probability reaches the threshold */
        MAX_PROB,
        /** stop when the normalised entropy is at most the threshold */
        ENTROPY,
        /** stop when the last {@code (int) threshold} exits agree */
        PATIENCE
    }

    /**
     * The outcome of an exit policy on a data set.
     *
     * @param accuracy the accuracy of the exits taken
     * @param meanMacs the mean multiply-accumulates per input
     * @param exitShare the share of inputs leaving at each exit
     */
    public record Outcome(double accuracy, double meanMacs, double[] exitShare) {
    }

    private final boolean conv;
    private final int[] widths;
    private final int classes;
    private final int side;
    private final List<Var> blockWeights = new ArrayList<>();
    private final List<Var> blockBiases = new ArrayList<>();
    private final List<Var> headWeights = new ArrayList<>();
    private final List<Var> headBiases = new ArrayList<>();
    private final long[] macs;

    private EarlyExitNetwork(boolean conv, int input, int side, int[] widths, int classes, long seed) {
        this.conv = conv;
        this.widths = widths.clone();
        this.classes = classes;
        this.side = side;
        SplittableRandom r = new SplittableRandom(seed);
        this.macs = new long[widths.length];
        long cum = 0;
        int in = input;
        for (int b = 0; b < widths.length; b++) {
            int out = widths[b];
            int size = conv ? sideAfter(b) : 0;
            if (conv) {
                float std = (float) Math.sqrt(2.0 / (in * 9));
                blockWeights.add(Var.param(Tensor.normal(Shape.of(out, in, 3, 3), 0, std, r)));
                blockBiases.add(Var.param(Tensor.zeros(Shape.of(out))));
                cum += (long) size * size * out * in * 9;
            } else {
                float std = (float) Math.sqrt(1.0 / in);
                blockWeights.add(Var.param(Tensor.normal(Shape.of(in, out), 0, std, r)));
                blockBiases.add(Var.param(Tensor.zeros(Shape.of(1, out))));
                cum += (long) in * out;
            }
            headWeights.add(Var.param(Tensor.normal(Shape.of(out, classes), 0, (float) Math.sqrt(1.0 / out), r)));
            headBiases.add(Var.param(Tensor.zeros(Shape.of(1, classes))));
            macs[b] = cum + (long) out * classes + (conv ? (long) size * size * out : 0);
            in = out;
        }
    }

    /**
     * A dense backbone with one exit per hidden layer.
     *
     * @param input the input size
     * @param widths the hidden sizes, one block (and exit) each
     * @param classes the number of classes
     * @param seed the seed
     * @return the network
     */
    public static EarlyExitNetwork mlp(int input, int[] widths, int classes, long seed) {
        return new EarlyExitNetwork(false, input, 0, widths, classes, seed);
    }

    /**
     * A convolutional backbone on {@code [n, channels, side, side]} images, one exit per block.
     *
     * @param channels the input channels
     * @param side the image side
     * @param widths the channels of each 3×3 block
     * @param classes the number of classes
     * @param seed the seed
     * @return the network
     */
    public static EarlyExitNetwork cnn(int channels, int side, int[] widths, int classes, long seed) {
        return new EarlyExitNetwork(true, channels, side, widths, classes, seed);
    }

    /** @return the number of exits */
    public int exits() {
        return widths.length;
    }

    /** @return the cumulative multiply-accumulates to reach and evaluate each exit */
    public long[] macs() {
        return macs.clone();
    }

    /** @return backbone parameters */
    public List<Var> backbone() {
        List<Var> p = new ArrayList<>();
        for (int b = 0; b < widths.length; b++) {
            p.add(blockWeights.get(b));
            p.add(blockBiases.get(b));
        }
        return p;
    }

    /**
     * @param exit an exit
     * @return that exit head's parameters
     */
    public List<Var> head(int exit) {
        return List.of(headWeights.get(exit), headBiases.get(exit));
    }

    /** @return all parameters */
    public List<Var> parameters() {
        List<Var> p = backbone();
        for (int b = 0; b < widths.length; b++) {
            p.addAll(head(b));
        }
        return p;
    }

    /** The feature-map side after block {@code b}: the first block keeps it, later ones halve it. */
    private int sideAfter(int b) {
        int s = side;
        for (int i = 1; i <= b; i++) {
            s = (s + 1) / 2;
        }
        return s;
    }

    private Var block(int b, Var h) {
        if (conv) {
            int[] one = {1, 1};
            int[] stride = b == 0 ? one : new int[] {2, 2};
            return Functional.conv(h, blockWeights.get(b), blockBiases.get(b), stride, one, one, one, 1).relu();
        }
        return h.matmul(blockWeights.get(b)).add(blockBiases.get(b)).tanh();
    }

    private Var headOf(int b, Var h) {
        Var f = conv ? Functional.globalPool(h, Functional.PoolKind.AVG, false) : h;
        return f.matmul(headWeights.get(b)).add(headBiases.get(b));
    }

    /**
     * The logits of every exit.
     *
     * @param x the inputs ({@code [n, features]} or {@code [n, c, side, side]})
     * @param detachBackbone whether the heads see detached features (frozen backbone)
     * @return one {@code [n, classes]} logit per exit
     */
    public Var[] logits(Var x, boolean detachBackbone) {
        Var[] out = new Var[widths.length];
        Var h = x;
        for (int b = 0; b < widths.length; b++) {
            h = block(b, h);
            out[b] = headOf(b, detachBackbone ? h.detach() : h);
        }
        return out;
    }

    /**
     * BranchyNet joint training with minibatch Adam.
     *
     * @param x the inputs
     * @param labels the classes
     * @param weights the loss weight of each exit
     * @param steps the steps
     * @param batch the minibatch size
     * @param learningRate the Adam step size
     * @param seed the minibatch seed
     */
    public void trainJoint(Tensor x, int[] labels, double[] weights, int steps, int batch, float learningRate,
            long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        List<Var> params = parameters();
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(x, labels, batch, r);
            Var[] l = logits(Var.of(b[0]), false);
            Var loss = null;
            for (int e = 0; e < l.length; e++) {
                Var ce = Functional.crossEntropy(l[e], b[1]).mul((float) weights[e]);
                loss = loss == null ? ce : loss.add(ce);
            }
            loss.backward();
            opt.step(params);
        }
    }

    /**
     * Shallow-deep training: the backbone with its final exit first, then the internal heads on
     * frozen features.
     *
     * @param x the inputs
     * @param labels the classes
     * @param steps the steps of each phase
     * @param batch the minibatch size
     * @param learningRate the Adam step size
     * @param seed the minibatch seed
     */
    public void trainFrozen(Tensor x, int[] labels, int steps, int batch, float learningRate, long seed) {
        int last = widths.length - 1;
        List<Var> first = backbone();
        first.addAll(head(last));
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(x, labels, batch, r);
            Var[] l = logits(Var.of(b[0]), false);
            Functional.crossEntropy(l[last], b[1]).backward();
            opt.step(first);
            for (Var p : parameters()) {
                p.zeroGrad();
            }
        }
        List<Var> heads = new ArrayList<>();
        for (int e = 0; e < last; e++) {
            heads.addAll(head(e));
        }
        Optimizer opt2 = Optimizer.adam(learningRate);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(x, labels, batch, r);
            Var[] l = logits(Var.of(b[0]), true);
            Var loss = null;
            for (int e = 0; e < last; e++) {
                Var ce = Functional.crossEntropy(l[e], b[1]);
                loss = loss == null ? ce : loss.add(ce);
            }
            if (loss != null) {
                loss.backward();
                opt2.step(heads);
            }
            for (Var p : parameters()) {
                p.zeroGrad();
            }
        }
    }

    private static Tensor[] batch(Tensor x, int[] labels, int size, SplittableRandom r) {
        int n = labels.length;
        int[] dims = x.shape().dims();
        int row = (int) (x.size() / n);
        float[] all = x.toArray();
        float[] bx = new float[size * row];
        float[] by = new float[size];
        for (int i = 0; i < size; i++) {
            int j = r.nextInt(n);
            System.arraycopy(all, j * row, bx, i * row, row);
            by[i] = labels[j];
        }
        int[] bd = dims.clone();
        bd[0] = size;
        return new Tensor[] {Tensor.of(Shape.of(bd), bx), Tensor.of(Shape.of(size), by)};
    }

    /**
     * Softmax probabilities of every exit.
     *
     * @param x the inputs
     * @param temperatures one temperature per exit (1 for none)
     * @return {@code [exit][n][classes]}
     */
    public double[][][] probabilities(Tensor x, double[] temperatures) {
        Var[] l = Var.noGrad(() -> logits(Var.of(x), false));
        int n = x.shape().dim(0);
        double[][][] out = new double[l.length][n][classes];
        for (int e = 0; e < l.length; e++) {
            float[] v = l[e].value().toArray();
            for (int i = 0; i < n; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < classes; c++) {
                    max = Math.max(max, v[i * classes + c] / temperatures[e]);
                }
                double z = 0;
                for (int c = 0; c < classes; c++) {
                    out[e][i][c] = Math.exp(v[i * classes + c] / temperatures[e] - max);
                    z += out[e][i][c];
                }
                for (int c = 0; c < classes; c++) {
                    out[e][i][c] /= z;
                }
            }
        }
        return out;
    }

    /**
     * The raw logits of every exit, for temperature fitting.
     *
     * @param x the inputs
     * @return {@code [exit][n][classes]}
     */
    public double[][][] logitArrays(Tensor x) {
        Var[] l = Var.noGrad(() -> logits(Var.of(x), false));
        int n = x.shape().dim(0);
        double[][][] out = new double[l.length][n][classes];
        for (int e = 0; e < l.length; e++) {
            float[] v = l[e].value().toArray();
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < classes; c++) {
                    out[e][i][c] = v[i * classes + c];
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ evaluation

    private static int argmax(double[] p) {
        int b = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[b]) {
                b = i;
            }
        }
        return b;
    }

    private static double confidence(double[] p, Rule rule) {
        if (rule == Rule.ENTROPY) {
            double h = 0;
            for (double v : p) {
                h -= v > 0 ? v * Math.log(v) : 0;
            }
            return -h / Math.log(p.length);
        }
        return p[argmax(p)];
    }

    /**
     * Runs an exit policy.
     *
     * @param probs {@code [exit][n][classes]}
     * @param labels the classes
     * @param rule the rule
     * @param threshold the max-probability threshold, the entropy bound (normalised to
     *     {@code [0, 1]}) or the patience
     * @param macs the cumulative MACs per exit
     * @return the outcome
     */
    public static Outcome evaluate(double[][][] probs, int[] labels, Rule rule, double threshold, long[] macs) {
        int exits = probs.length;
        int n = labels.length;
        int right = 0;
        double cost = 0;
        double[] share = new double[exits];
        for (int i = 0; i < n; i++) {
            int stop = exits - 1;
            int streak = 0;
            int previous = -1;
            for (int e = 0; e < exits; e++) {
                double[] p = probs[e][i];
                boolean go;
                if (rule == Rule.PATIENCE) {
                    int y = argmax(p);
                    streak = y == previous ? streak + 1 : 1;
                    previous = y;
                    go = streak >= (int) threshold;
                } else if (rule == Rule.ENTROPY) {
                    go = -confidence(p, rule) <= threshold;
                } else {
                    go = confidence(p, rule) >= threshold;
                }
                if (go) {
                    stop = e;
                    break;
                }
            }
            right += argmax(probs[stop][i]) == labels[i] ? 1 : 0;
            cost += macs[stop];
            share[stop] += 1.0 / n;
        }
        return new Outcome(right / (double) n, cost / n, share);
    }

    /**
     * The area under the accuracy-compute curve of the max-probability rule, swept over
     * thresholds, with compute normalised by the full network's MACs (the curve is extended flat
     * to the ends).
     *
     * @param probs {@code [exit][n][classes]}
     * @param labels the classes
     * @param macs cumulative MACs per exit
     * @return the area, in {@code [0, 1]}
     */
    public static double accuracyComputeArea(double[][][] probs, int[] labels, long[] macs) {
        double full = macs[macs.length - 1];
        List<double[]> pts = new ArrayList<>();
        for (int k = 0; k <= 40; k++) {
            double t = k == 40 ? 1.01 : k / 40.0;
            Outcome o = evaluate(probs, labels, Rule.MAX_PROB, t, macs);
            pts.add(new double[] {o.meanMacs() / full, o.accuracy()});
        }
        pts.sort((a, b) -> Double.compare(a[0], b[0]));
        // keep the upper envelope: the best accuracy at or below each compute
        double area = pts.get(0)[1] * pts.get(0)[0];
        double best = pts.get(0)[1];
        for (int i = 1; i < pts.size(); i++) {
            double[] a = pts.get(i - 1);
            double[] b = pts.get(i);
            double nb = Math.max(best, b[1]);
            area += (b[0] - a[0]) * (best + nb) / 2;
            best = nb;
        }
        area += (1 - pts.get(pts.size() - 1)[0]) * best;
        return area;
    }

    /**
     * The overthinking rate: inputs that some earlier exit classifies correctly while the final
     * exit is wrong.
     *
     * @param probs {@code [exit][n][classes]}
     * @param labels the classes
     * @return the share of inputs
     */
    public static double overthinking(double[][][] probs, int[] labels) {
        int last = probs.length - 1;
        int count = 0;
        for (int i = 0; i < labels.length; i++) {
            if (argmax(probs[last][i]) == labels[i]) {
                continue;
            }
            for (int e = 0; e < last; e++) {
                if (argmax(probs[e][i]) == labels[i]) {
                    count++;
                    break;
                }
            }
        }
        return count / (double) labels.length;
    }

    /**
     * EEFP of one exit: the AUROC of its maximum probability against the target "should stop",
     * true when the exit is right or no later exit is right.
     *
     * @param probs {@code [exit][n][classes]}
     * @param labels the classes
     * @param exit the exit
     * @return the AUROC, or NaN if one class of the target is empty
     */
    public static double eefp(double[][][] probs, int[] labels, int exit) {
        int n = labels.length;
        double[] score = new double[n];
        boolean[] stop = new boolean[n];
        for (int i = 0; i < n; i++) {
            score[i] = confidence(probs[exit][i], Rule.MAX_PROB);
            boolean right = argmax(probs[exit][i]) == labels[i];
            boolean laterRight = false;
            for (int e = exit + 1; e < probs.length; e++) {
                laterRight |= argmax(probs[e][i]) == labels[i];
            }
            stop[i] = right || !laterRight;
        }
        return auroc(score, stop);
    }

    /**
     * The mean EEFP over the internal exits.
     *
     * @param probs {@code [exit][n][classes]}
     * @param labels the classes
     * @return the mean over exits whose AUROC is defined
     */
    public static double meanEefp(double[][][] probs, int[] labels) {
        double s = 0;
        int k = 0;
        for (int e = 0; e < probs.length - 1; e++) {
            double v = eefp(probs, labels, e);
            if (!Double.isNaN(v)) {
                s += v;
                k++;
            }
        }
        return k == 0 ? Double.NaN : s / k;
    }

    private static double auroc(double[] score, boolean[] positive) {
        Integer[] idx = new Integer[score.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(score[a], score[b]));
        double rankSum = 0;
        int pos = 0;
        int i = 0;
        while (i < idx.length) {
            int j = i;
            while (j + 1 < idx.length && score[idx[j + 1]] == score[idx[i]]) {
                j++;
            }
            double rank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) {
                if (positive[idx[k]]) {
                    rankSum += rank;
                    pos++;
                }
            }
            i = j + 1;
        }
        int neg = idx.length - pos;
        if (pos == 0 || neg == 0) {
            return Double.NaN;
        }
        return (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }
}
