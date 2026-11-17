package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Attribution patching (AtP) and AtP*: fast, gradient-based estimates of every node's
 * activation-patching effect in a {@link HookedDecoder}. A node is one component's output
 * (attention or feed-forward) of one block at one position, i.e. one term added to the
 * residual stream. Exact activation patching ({@link #exact}) runs the corrupted prompt once
 * per node with that node's clean value copied in; AtP replaces all of those runs with a
 * first-order Taylor expansion around the corrupted run,
 *
 * <pre>c(n) ~ (a_clean(n) - a_corrupt(n)) . d metric / d a(n) |_corrupt</pre>
 *
 * <p>which needs one clean forward, one corrupted forward and one backward pass for all
 * nodes at once ({@link #atp}).
 *
 * <p>AtP* adds the paper's GradDrop fix against cancellation: when paths through a later
 * layer cancel a node's effect in the gradient, the node looks unimportant though patching
 * it matters. Re-running the backward pass with the gradient through each layer's residual
 * contribution blocked in turn and keeping the largest absolute estimate exposes such nodes
 * ({@link #atpStar}). The paper's other fix (QK-fix: recomputing the attention softmax
 * exactly for key/query nodes) needs per-head attention-score nodes, which
 * {@code HookedDecoder} does not expose; here the nodes are whole component outputs, for
 * which the linearisation through attention is the plain one.
 *
 * <p>Nanda 2023, "Attribution Patching"; Kramár et al. 2024, "AtP*: An efficient and
 * scalable method for localizing LLM behaviour to components",
 * <a href="https://arxiv.org/abs/2403.00745">arXiv:2403.00745</a>.
 *
 * <p>The decoder must have no adapters of its own: they are used as the hooks and cleared
 * afterwards.
 */
public final class AttributionPatching {

    /** Reads a scalar metric (such as a logit difference) from the logits. */
    @FunctionalInterface
    public interface Metric {
        /**
         * @param logits {@code [batch, steps, vocab]}
         * @return a scalar
         */
        Var apply(Var logits);
    }

    /** Which half of a block a node is. */
    public enum Component {
        /** the attention output */
        ATTENTION,
        /** the feed-forward output */
        MLP
    }

    /**
     * A node.
     *
     * @param block the block
     * @param component attention or feed-forward
     * @param position the position
     */
    public record Node(int block, Component component, int position) {
    }

    private AttributionPatching() {
    }

    /**
     * Every node of a decoder for sequences of a given length, in the order the effect
     * arrays use: block, then component, then position.
     *
     * @param model the decoder
     * @param steps the sequence length
     * @return the nodes
     */
    public static List<Node> nodes(HookedDecoder model, int steps) {
        List<Node> out = new ArrayList<>();
        for (int b = 0; b < model.layers(); b++) {
            for (Component c : Component.values()) {
                for (int t = 0; t < steps; t++) {
                    out.add(new Node(b, c, t));
                }
            }
        }
        return out;
    }

    /** An adapter that records what passes and can patch, perturb or cut the gradient. */
    private static final class Hook implements Layer {
        Tensor recorded;
        Tensor patchValue;
        Tensor patchMask;
        Var delta;
        boolean dropGradient;

        @Override
        public Var forward(Var input) {
            recorded = input.value();
            Var x = dropGradient ? input.detach() : input;
            if (patchValue != null) {
                x = x.mul(patchMask.neg().add(1f)).add(patchValue.mul(patchMask));
            }
            return delta == null ? x : x.add(delta);
        }
    }

    private static Hook[][] install(HookedDecoder model) {
        Hook[][] hooks = new Hook[model.layers()][2];
        for (int b = 0; b < model.layers(); b++) {
            hooks[b][0] = new Hook();
            hooks[b][1] = new Hook();
            model.adapters(b, hooks[b][0], hooks[b][1]);
        }
        return hooks;
    }

    private static void remove(HookedDecoder model) {
        for (int b = 0; b < model.layers(); b++) {
            model.adapters(b, null, null);
        }
    }

    private static double value(HookedDecoder model, int[][] tokens, Metric metric) {
        return metric.apply(model.forward(tokens).logits()).value().item();
    }

    /**
     * Exact activation patching: the change of the metric when one node of the corrupted run
     * takes its clean value, for every node.
     *
     * @param model the decoder
     * @param clean clean prompts {@code [batch][steps]}
     * @param corrupted corrupted prompts of the same shape
     * @param metric the metric
     * @return {@code metric(patched) - metric(corrupted)} per node, in {@link #nodes} order
     */
    public static double[] exact(HookedDecoder model, int[][] clean, int[][] corrupted, Metric metric) {
        Hook[][] hooks = install(model);
        try {
            model.forward(clean);
            Tensor[][] cleanValues = recorded(hooks);
            double base = value(model, corrupted, metric);
            int steps = clean[0].length;
            List<Node> nodes = nodes(model, steps);
            double[] out = new double[nodes.size()];
            for (int i = 0; i < out.length; i++) {
                Node n = nodes.get(i);
                Hook h = hooks[n.block()][n.component().ordinal()];
                Tensor v = cleanValues[n.block()][n.component().ordinal()];
                h.patchValue = v;
                h.patchMask = positionMask(v.shape(), n.position());
                out[i] = value(model, corrupted, metric) - base;
                h.patchValue = null;
                h.patchMask = null;
            }
            return out;
        } finally {
            remove(model);
        }
    }

    /**
     * Attribution patching: the first-order estimate of {@link #exact} from one backward pass.
     *
     * @param model the decoder
     * @param clean clean prompts
     * @param corrupted corrupted prompts
     * @param metric the metric
     * @return the estimate per node, in {@link #nodes} order
     */
    public static double[] atp(HookedDecoder model, int[][] clean, int[][] corrupted, Metric metric) {
        return estimate(model, clean, corrupted, metric, -1);
    }

    /**
     * AtP* with GradDrop: for every block {@code l}, the AtP estimate with the gradient
     * through block {@code l}'s residual contributions blocked; the score of a node is its
     * largest absolute estimate over those runs and the plain one, so a node whose effect
     * cancels in the plain gradient still shows up.
     *
     * @param model the decoder
     * @param clean clean prompts
     * @param corrupted corrupted prompts
     * @param metric the metric
     * @return a non-negative importance score per node, in {@link #nodes} order
     */
    public static double[] atpStar(HookedDecoder model, int[][] clean, int[][] corrupted, Metric metric) {
        double[] plain = estimate(model, clean, corrupted, metric, -1);
        double[] out = new double[plain.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.abs(plain[i]);
        }
        for (int l = 0; l < model.layers(); l++) {
            double[] dropped = estimate(model, clean, corrupted, metric, l);
            for (int i = 0; i < out.length; i++) {
                out[i] = Math.max(out[i], Math.abs(dropped[i]));
            }
        }
        return out;
    }

    private static double[] estimate(HookedDecoder model, int[][] clean, int[][] corrupted, Metric metric,
            int dropBlock) {
        Hook[][] hooks = install(model);
        try {
            model.forward(clean);
            Tensor[][] cleanValues = recorded(hooks);
            List<Var> deltas = new ArrayList<>();
            for (int b = 0; b < hooks.length; b++) {
                for (Hook h : hooks[b]) {
                    h.delta = Var.param(Tensor.zeros(cleanValues[b][0].shape()));
                    h.dropGradient = b == dropBlock;
                    deltas.add(h.delta);
                }
            }
            Var m = metric.apply(model.forward(corrupted).logits());
            Tensor[][] corruptValues = recorded(hooks);
            List<Var> grads = Var.gradients(m, deltas, false);
            int steps = clean[0].length;
            List<Node> nodes = nodes(model, steps);
            double[] out = new double[nodes.size()];
            for (int i = 0; i < out.length; i++) {
                Node n = nodes.get(i);
                int c = n.component().ordinal();
                Var g = grads.get(n.block() * 2 + c);
                if (g == null) {
                    continue;
                }
                Tensor diff = cleanValues[n.block()][c].sub(corruptValues[n.block()][c]);
                out[i] = diff.mul(g.value()).mul(positionMask(diff.shape(), n.position())).sum().item();
            }
            return out;
        } finally {
            remove(model);
        }
    }

    private static Tensor[][] recorded(Hook[][] hooks) {
        Tensor[][] out = new Tensor[hooks.length][2];
        for (int b = 0; b < hooks.length; b++) {
            out[b][0] = hooks[b][0].recorded;
            out[b][1] = hooks[b][1].recorded;
        }
        return out;
    }

    private static Tensor positionMask(Shape shape, int position) {
        int batch = shape.dim(0);
        int steps = shape.dim(1);
        int dim = shape.dim(2);
        float[] m = new float[batch * steps * dim];
        for (int b = 0; b < batch; b++) {
            for (int d = 0; d < dim; d++) {
                m[(b * steps + position) * dim + d] = 1;
            }
        }
        return Tensor.of(shape, m);
    }

    /**
     * Spearman's rank correlation of two score vectors.
     *
     * @param a the first scores
     * @param b the second scores
     * @return the correlation of their ranks
     */
    public static double spearman(double[] a, double[] b) {
        double[] ra = ranks(a);
        double[] rb = ranks(b);
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < a.length; i++) {
            ma += ra[i] / a.length;
            mb += rb[i] / a.length;
        }
        double num = 0;
        double da = 0;
        double db = 0;
        for (int i = 0; i < a.length; i++) {
            num += (ra[i] - ma) * (rb[i] - mb);
            da += (ra[i] - ma) * (ra[i] - ma);
            db += (rb[i] - mb) * (rb[i] - mb);
        }
        return num / Math.sqrt(Math.max(da * db, 1e-300));
    }

    private static double[] ranks(double[] x) {
        Integer[] idx = new Integer[x.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        java.util.Arrays.sort(idx, (p, q) -> Double.compare(x[p], x[q]));
        double[] r = new double[x.length];
        int i = 0;
        while (i < idx.length) {
            int j = i;
            while (j + 1 < idx.length && x[idx[j + 1]] == x[idx[i]]) {
                j++;
            }
            for (int k = i; k <= j; k++) {
                r[idx[k]] = (i + j) / 2.0;
            }
            i = j + 1;
        }
        return r;
    }
}
