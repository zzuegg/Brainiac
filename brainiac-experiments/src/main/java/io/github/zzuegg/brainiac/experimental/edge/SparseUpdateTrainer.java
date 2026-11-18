package io.github.zzuegg.brainiac.experimental.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * On-device training of an {@link EdgeNet} under a memory budget: a sparse update (a chosen set
 * of biases and of layers' output channels is trained, everything else is frozen), with the
 * weights held as int8 and updated with quantisation-aware scaling (QAS) and stochastic rounding.
 *
 * <p>With per-channel weight scale {@code s}, the int8 weight {@code q = w / s} receives the
 * gradient {@code s g_w}, so its gradient-to-weight ratio is {@code s^2} times that of the float
 * weight and plain SGD on int8 weights barely moves them. QAS multiplies the int8 gradient by
 * {@code s^-2}, which restores the float update {@code s dq = -lr g_w}. Contribution analysis
 * measures how much each candidate (a layer's weights at a channel ratio, or the biases of the
 * last layers) improves accuracy when fine-tuned alone, and a greedy knapsack picks the
 * candidates under the training-memory budget from {@link MemoryPlanner#trainingMemory}.
 *
 * <p>Activations stay float here (the int8 forward graph is {@link IntegerInference}'s job).
 *
 * <p>Paper: Lin et al., "On-Device Training Under 256KB Memory", NeurIPS 2022,
 * <a href="https://arxiv.org/abs/2206.15472">arXiv:2206.15472</a>.
 */
public final class SparseUpdateTrainer {

    /** Number format of the updated weights. */
    public enum Precision {
        /** float32 weights, plain SGD */
        FLOAT,
        /** int8 weights, SGD on the int8 values with stochastic rounding, no scaling */
        INT8,
        /** int8 weights with quantisation-aware scaling */
        INT8_QAS
    }

    /**
     * What is updated.
     *
     * @param weightShare per unit, the share of output channels whose weights are trained
     * @param bias per unit, whether its bias is trained
     */
    public record Selection(double[] weightShare, boolean[] bias) {

        /**
         * Nothing selected.
         *
         * @param units the network's units
         * @return an empty selection
         */
        public static Selection none(int units) {
            return new Selection(new double[units], new boolean[units]);
        }

        /** @return a copy */
        public Selection copy() {
            return new Selection(weightShare.clone(), bias.clone());
        }

        @Override
        public String toString() {
            StringBuilder b = new StringBuilder();
            for (int i = 0; i < weightShare.length; i++) {
                if (weightShare[i] > 0) {
                    b.append("W").append(i).append('@').append(weightShare[i]).append(' ');
                }
                if (bias[i]) {
                    b.append("b").append(i).append(' ');
                }
            }
            return b.toString().trim();
        }
    }

    private final EdgeNet net;
    private final Precision precision;
    private final float lr;
    private final List<Tensor> snapshot = new ArrayList<>();
    private final float[][] scales;
    private final float[][] ints;

    /**
     * Prepares a pretrained network: folds batch norm, and for int8 precision rounds every weight
     * to its per-channel int8 grid. The state at this point is what {@link #reset} restores.
     *
     * @param net the network
     * @param precision the weight format
     * @param lr the SGD learning rate (in float-weight units)
     */
    public SparseUpdateTrainer(EdgeNet net, Precision precision, float lr) {
        this.net = net.foldBatchNorm();
        this.precision = precision;
        this.lr = lr;
        int n = net.units().size();
        scales = new float[n][];
        ints = new float[n][];
        for (int i = 0; i < n; i++) {
            Var w = weight(net.units().get(i));
            if (w != null && precision != Precision.FLOAT) {
                boolean conv = net.units().get(i) instanceof EdgeNet.Conv;
                scales[i] = IntegerInference.weightScales(w.value(), conv);
                w.assign(IntegerInference.fakeQuantWeights(w.value(), conv));
            }
        }
        for (Var p : net.parameters()) {
            snapshot.add(p.value());
        }
        reset();
    }

    private static Var weight(EdgeNet.Unit u) {
        return switch (u) {
            case EdgeNet.Conv c -> c.weight();
            case EdgeNet.Linear l -> l.weight();
            case EdgeNet.Pool p -> null;
        };
    }

    private static Var bias(EdgeNet.Unit u) {
        return switch (u) {
            case EdgeNet.Conv c -> c.bias();
            case EdgeNet.Linear l -> l.bias();
            case EdgeNet.Pool p -> null;
        };
    }

    /** Restores the pretrained (and, for int8, rounded) state. */
    public void reset() {
        List<Var> p = net.parameters();
        for (int i = 0; i < p.size(); i++) {
            p.get(i).assign(snapshot.get(i));
        }
        for (int i = 0; i < ints.length; i++) {
            if (scales[i] != null) {
                Var w = weight(net.units().get(i));
                boolean conv = net.units().get(i) instanceof EdgeNet.Conv;
                float[] v = w.value().toArray();
                ints[i] = new float[v.length];
                for (int j = 0; j < v.length; j++) {
                    ints[i][j] = Math.round(v[j] / scales[i][channel(j, v.length, scales[i].length, conv)]);
                }
            }
        }
    }

    private static int channel(int j, int n, int out, boolean conv) {
        return conv ? j / (n / out) : j % out;
    }

    /** @return the network */
    public EdgeNet net() {
        return net;
    }

    /**
     * The output channels updated at a share: those with the largest weight L2 norms.
     *
     * @param unit the unit
     * @param share the share
     * @return per output channel, whether it is updated
     */
    public boolean[] channels(int unit, double share) {
        EdgeNet.Unit u = net.units().get(unit);
        Var w = weight(u);
        boolean conv = u instanceof EdgeNet.Conv;
        float[] v = w.value().toArray();
        int out = conv ? w.shape().dim(0) : w.shape().dim(1);
        double[] norm = new double[out];
        for (int j = 0; j < v.length; j++) {
            norm[channel(j, v.length, out, conv)] += v[j] * v[j];
        }
        Integer[] order = new Integer[out];
        for (int i = 0; i < out; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(norm[b], norm[a]));
        boolean[] on = new boolean[out];
        int k = (int) Math.ceil(share * out);
        for (int i = 0; i < k; i++) {
            on[order[i]] = true;
        }
        return on;
    }

    /**
     * Trains a selection with SGD on random mini-batches.
     *
     * @param x inputs
     * @param y class ids
     * @param sel what to update
     * @param extra further trainable parameters (adapters, side branches; plain float SGD)
     * @param steps steps
     * @param batch batch size
     * @param seed the seed
     * @return the last loss
     */
    public float train(Tensor x, Tensor y, Selection sel, List<Var> extra, int steps, int batch, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = net.units().size();
        boolean[][] masks = new boolean[n][];
        for (int i = 0; i < n; i++) {
            if (sel.weightShare()[i] > 0) {
                masks[i] = channels(i, sel.weightShare()[i]);
            }
        }
        List<Var> all = new ArrayList<>(net.parameters());
        all.addAll(extra);
        int rows = x.shape().dim(0);
        float last = 0;
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(Math.max(1, rows - batch));
            int to = Math.min(rows, from + batch);
            all.forEach(Var::zeroGrad);
            Var loss = Functional.crossEntropy(net.forward(Var.of(x.slice(0, from, to)), true), y.slice(0, from, to));
            last = loss.value().item();
            loss.backward();
            for (int i = 0; i < n; i++) {
                EdgeNet.Unit u = net.units().get(i);
                if (masks[i] != null) {
                    updateWeight(i, u, masks[i], r);
                }
                if (sel.bias()[i]) {
                    sgd(bias(u));
                }
            }
            for (Var e : extra) {
                sgd(e);
            }
            all.forEach(Var::zeroGrad);
        }
        return last;
    }

    private void sgd(Var p) {
        Tensor g = p.grad();
        if (g == null) {
            return;
        }
        float[] v = p.value().toArray();
        float[] gv = g.toArray();
        for (int j = 0; j < v.length; j++) {
            v[j] -= lr * gv[j];
        }
        p.assign(Tensor.of(p.shape(), v));
    }

    private void updateWeight(int i, EdgeNet.Unit u, boolean[] mask, SplittableRandom r) {
        Var w = weight(u);
        Tensor g = w.grad();
        if (g == null) {
            return;
        }
        boolean conv = u instanceof EdgeNet.Conv;
        float[] v = w.value().toArray();
        float[] gv = g.toArray();
        int out = mask.length;
        for (int j = 0; j < v.length; j++) {
            int c = channel(j, v.length, out, conv);
            if (!mask[c]) {
                continue;
            }
            if (precision == Precision.FLOAT) {
                v[j] -= lr * gv[j];
            } else {
                float s = scales[i][c];
                float gq = s * gv[j];                          // gradient of the int8 value
                if (precision == Precision.INT8_QAS) {
                    gq /= s * s;                               // quantisation-aware scaling
                }
                float q = ints[i][j] - lr * gq;
                float fl = (float) Math.floor(q);
                q = fl + (r.nextDouble() < q - fl ? 1 : 0);    // stochastic rounding
                q = Math.max(-127, Math.min(127, q));
                ints[i][j] = q;
                v[j] = q * s;
            }
        }
        w.assign(Tensor.of(w.shape(), v));
    }

    /**
     * Training memory of a selection (int8 activations).
     *
     * @param sel the selection
     * @param extraStored further stored activation bytes
     * @param extraParameters further trained parameters
     * @return the memory
     */
    public MemoryPlanner.TrainingMemory memory(Selection sel, long extraStored, long extraParameters) {
        return MemoryPlanner.trainingMemory(net, sel.weightShare(), sel.bias(), extraStored, extraParameters, 1);
    }

    /**
     * Contribution analysis: from the pretrained state, fine-tunes the last unit plus one
     * candidate at a time and records the validation accuracy gain over fine-tuning the last unit
     * alone. Candidates are each earlier unit's weights at each channel share, and the biases of
     * the last {@code k} units.
     *
     * @param x training inputs
     * @param y training labels
     * @param vx validation inputs
     * @param vy validation labels
     * @param shares channel shares to try
     * @param steps steps per trial
     * @param seed the seed
     * @return the candidates with their gains
     */
    public List<Candidate> contributions(Tensor x, Tensor y, Tensor vx, Tensor vy, double[] shares, int steps,
            long seed) {
        int n = net.units().size();
        int head = n - 1;
        Selection base = Selection.none(n);
        base.weightShare()[head] = 1;
        base.bias()[head] = true;
        reset();
        train(x, y, base, List.of(), steps, 32, seed);
        double baseAcc = net.accuracy(vx, vy);
        List<Candidate> out = new ArrayList<>();
        for (int i = 0; i < head; i++) {
            if (weight(net.units().get(i)) == null) {
                continue;
            }
            for (double s : shares) {
                Selection sel = base.copy();
                sel.weightShare()[i] = s;
                out.add(trial(sel, "W" + i + "@" + s, x, y, vx, vy, steps, seed, baseAcc));
            }
        }
        for (int k = 2; k <= n; k++) {
            Selection sel = base.copy();
            for (int i = n - k; i < n; i++) {
                sel.bias()[i] = weight(net.units().get(i)) != null;
            }
            out.add(trial(sel, "bias last " + k, x, y, vx, vy, steps, seed, baseAcc));
        }
        reset();
        return out;
    }

    private Candidate trial(Selection sel, String name, Tensor x, Tensor y, Tensor vx, Tensor vy, int steps, long seed,
            double baseAcc) {
        reset();
        train(x, y, sel, List.of(), steps, 32, seed);
        return new Candidate(name, sel, net.accuracy(vx, vy) - baseAcc, memory(sel, 0, 0).total());
    }

    /**
     * A contribution-analysis candidate.
     *
     * @param name a label
     * @param selection the selection it adds to the last unit
     * @param gain validation accuracy gain over the last unit alone
     * @param bytes its training memory
     */
    public record Candidate(String name, Selection selection, double gain, long bytes) {
    }

    /**
     * Greedy knapsack over candidates: the union of candidates in order of gain is kept while the
     * training memory of the union stays within the budget.
     *
     * @param candidates analysed candidates
     * @param budgetBytes the training-memory budget
     * @return the selection
     */
    public Selection select(List<Candidate> candidates, long budgetBytes) {
        int n = net.units().size();
        Selection sel = Selection.none(n);
        sel.weightShare()[n - 1] = 1;
        sel.bias()[n - 1] = true;
        List<Candidate> sorted = new ArrayList<>(candidates);
        sorted.sort((a, b) -> Double.compare(b.gain(), a.gain()));
        for (Candidate c : sorted) {
            if (c.gain() <= 0) {
                break;
            }
            Selection t = sel.copy();
            for (int i = 0; i < n; i++) {
                t.weightShare()[i] = Math.max(t.weightShare()[i], c.selection().weightShare()[i]);
                t.bias()[i] |= c.selection().bias()[i];
            }
            if (memory(t, 0, 0).total() <= budgetBytes) {
                sel = t;
            }
        }
        return sel;
    }
}
