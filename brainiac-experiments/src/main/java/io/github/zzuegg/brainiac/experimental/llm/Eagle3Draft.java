package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * EAGLE-3 speculative decoding, with P-EAGLE parallel drafting. Two changes over
 * {@link EagleDraft}:
 * <ul>
 * <li><b>Multi-layer feature fusion.</b> The draft head starts from a learned fusion of the
 *     target's low-, mid- and high-level features (the residual stream after an early and a
 *     late layer, and the final hidden state, read with {@link PagedLlama#tap}) instead of the
 *     top feature alone, which is shaped for predicting one token, not several.</li>
 * <li><b>Training-time test.</b> The head is trained on its own rollouts: the first step
 *     starts from the fused target feature, every later step from the head's own previous
 *     output, exactly as at inference, with a token loss at each step (EAGLE-3 drops the
 *     feature-regression loss, so the head's state need not imitate the target's).</li>
 * </ul>
 * A drafted tree is verified in one {@link PagedLlama} pass with tree attention, as in
 * {@link EagleDraft}; the output equals the target's greedy decoding exactly.
 *
 * <p><b>Parallel drafting (P-EAGLE).</b> Drafting K tokens one after another costs K draft
 * passes. {@link #generateParallel} instead predicts all K tokens in <em>one</em> draft pass:
 * depth {@code k} reads the fused feature plus a learned depth embedding in place of the
 * still-unknown intermediate tokens. The depth embeddings train jointly with the sequential
 * head ({@link #train} optimizes both).
 *
 * <p>Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via
 * Training-Time Test" (2025), <a href="https://arxiv.org/abs/2503.01840">arXiv:2503.01840</a>;
 * vLLM, "P-EAGLE: parallel drafting" (2025), <a href="https://vllm.ai/blog/p-eagle">vllm.ai/blog/p-eagle</a>.
 */
public final class Eagle3Draft {

    /**
     * A generation.
     *
     * @param tokens the new tokens, equal to the target's greedy decoding
     * @param passes the target passes run
     * @param draftPasses the draft-head passes run
     * @param accepted the draft tokens kept
     * @param millis the wall time
     */
    public record Result(int[] tokens, int passes, int draftPasses, int accepted, double millis) {

        /** @return new tokens per target pass (the mean acceptance length plus one) */
        public double tokensPerPass() {
            return tokens.length / (double) passes;
        }
    }

    private final PagedLlama target;
    private final int hidden;
    private final int[] taps;
    private final int unroll;
    private final Var fuse;
    private final Var fuseBias;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var depth;
    private final Var p1;
    private final Var pb1;
    private final Var p2;
    private final Var pb2;
    private float featureWeight;

    /**
     * @param target the target model; its weights stay frozen
     * @param unroll the training-time-test steps, also the most tokens drafted in parallel
     * @param seed seeds the head's weights
     */
    public Eagle3Draft(PagedLlama target, int unroll, long seed) {
        if (unroll < 1) {
            throw new IllegalArgumentException("unroll must be at least 1");
        }
        this.target = target;
        this.hidden = target.hiddenSize();
        this.unroll = unroll;
        int layers = target.model().config().layers();
        this.taps = new int[] {0, layers > 2 ? layers / 2 : layers - 1};
        SplittableRandom r = new SplittableRandom(seed);
        int h = hidden;
        // a residual fusion that starts at zero: the head begins exactly where EAGLE does
        fuse = Var.param(Tensor.zeros(Shape.of(3 * h, h)));
        fuseBias = Var.param(Tensor.zeros(Shape.of(h)));
        w1 = Var.param(Tensor.normal(Shape.of(2 * h, 2 * h), 0, (float) Math.sqrt(1.0 / (2 * h)), r));
        b1 = Var.param(Tensor.zeros(Shape.of(2 * h)));
        w2 = Var.param(Tensor.normal(Shape.of(2 * h, h), 0, (float) Math.sqrt(0.5 / (2 * h)), r));
        b2 = Var.param(Tensor.zeros(Shape.of(h)));
        depth = Var.param(Tensor.normal(Shape.of(unroll, h), 0, 0.02f, r));
        p1 = Var.param(Tensor.normal(Shape.of(2 * h, 2 * h), 0, (float) Math.sqrt(1.0 / (2 * h)), r));
        pb1 = Var.param(Tensor.zeros(Shape.of(2 * h)));
        p2 = Var.param(Tensor.normal(Shape.of(2 * h, h), 0, (float) Math.sqrt(0.5 / (2 * h)), r));
        pb2 = Var.param(Tensor.zeros(Shape.of(h)));
    }

    /**
     * Adds EAGLE's feature regression on the first draft step (the predicted state against the
     * target's next final hidden state). EAGLE-3 drops it to scale with data; on small data it
     * still speeds up learning.
     *
     * @param weight the regression weight, 0 (the default) for pure EAGLE-3
     * @return this head
     */
    public Eagle3Draft featureLoss(float weight) {
        this.featureWeight = weight;
        return this;
    }

    /** @return the head's trainable weights */
    public List<Var> parameters() {
        return List.of(fuse, fuseBias, w1, b1, w2, b2, depth, p1, pb1, p2, pb2);
    }

    /** @return the layers whose residual stream is fused with the final hidden state */
    public int[] tappedLayers() {
        return taps.clone();
    }

    /** {@code f + MLP([f, e])}. */
    private Var predict(Var features, Var embeddings) {
        Var x = Var.concat(1, features, embeddings);
        Var h = x.matmul(w1).add(b1).activation(ActivationOp.SILU, 0);
        return features.add(h.matmul(w2).add(b2));
    }

    /** The parallel head: its own MLP over the depth-shifted fused feature and the pending token. */
    private Var predictParallel(Var features, Var embeddings) {
        Var x = Var.concat(1, features, embeddings);
        Var h = x.matmul(p1).add(pb1).activation(ActivationOp.SILU, 0);
        return features.add(h.matmul(p2).add(pb2));
    }

    /** The fused feature of rows {@code [low, mid, high]}. */
    private Var fused(Var low, Var mid, Var high) {
        return high.add(Var.concat(1, rms(low), rms(mid), high).matmul(fuse).add(fuseBias));
    }

    /** Residual-stream rows scaled to unit RMS, like the final hidden state the head also reads. */
    private static Var rms(Var x) {
        return x.div(x.square().mean(1, true).add(1e-6f).sqrt());
    }

    /** One draft step's input rows for the parallel head at depth {@code k} (0-based). */
    private Var parallelInput(Var f0, int k, int rows) {
        return f0.add(depth.slice(0, k, k + 1).expand(Shape.of(rows, hidden)));
    }

    /**
     * Trains the fusion, the head and the parallel depth embeddings on the target's own
     * features, with training-time test over {@code unroll} steps.
     *
     * @param sequences token sequences (for example the target's greedy continuations)
     * @param steps optimizer steps, each over every position
     * @param optimizer the optimizer
     * @return the last loss
     */
    public float train(int[][] sequences, int steps, Optimizer optimizer) {
        KvBlockStore store = target.store();
        target.tap(taps);
        List<float[]> lows = new ArrayList<>();
        List<float[]> mids = new ArrayList<>();
        List<float[]> highs = new ArrayList<>();
        List<float[]> nextHighs = new ArrayList<>();
        List<int[]> inputs = new ArrayList<>();    // tokens x_(t+1) .. x_(t+unroll)
        List<int[]> labels = new ArrayList<>();    // tokens x_(t+2) .. x_(t+1+unroll)
        for (int[] s : sequences) {
            int seq = store.newSequence();
            PagedLlama.Output out = target.forward(List.of(PagedLlama.Pass.chain(seq, s)), true).getFirst();
            store.free(seq);
            float[] low = target.tapped(0);
            float[] mid = target.tapped(1);
            for (int t = 0; t + 1 + unroll < s.length; t++) {
                lows.add(Arrays.copyOfRange(low, t * hidden, (t + 1) * hidden));
                mids.add(Arrays.copyOfRange(mid, t * hidden, (t + 1) * hidden));
                highs.add(Arrays.copyOfRange(out.hidden(), t * hidden, (t + 1) * hidden));
                nextHighs.add(Arrays.copyOfRange(out.hidden(), (t + 1) * hidden, (t + 2) * hidden));
                inputs.add(Arrays.copyOfRange(s, t + 1, t + 1 + unroll));
                labels.add(Arrays.copyOfRange(s, t + 2, t + 2 + unroll));
            }
        }
        target.tap();
        int n = lows.size();
        Var low = Var.of(Tensor.of(Shape.of(n, hidden), flatten(lows)));
        Var mid = Var.of(Tensor.of(Shape.of(n, hidden), flatten(mids)));
        Var high = Var.of(Tensor.of(Shape.of(n, hidden), flatten(highs)));
        Var nextHigh = Var.of(Tensor.of(Shape.of(n, hidden), flatten(nextHighs)));
        Var[] emb = new Var[unroll];
        Tensor[] want = new Tensor[unroll];
        for (int k = 0; k < unroll; k++) {
            int[] toks = new int[n];
            float[] ids = new float[n];
            for (int i = 0; i < n; i++) {
                toks[i] = inputs.get(i)[k];
                ids[i] = labels.get(i)[k];
            }
            emb[k] = Var.of(Tensor.of(Shape.of(n, hidden), target.embed(toks)));
            want[k] = Tensor.of(Shape.of(n), ids);
        }
        Var head = Var.of(target.outputWeights());
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            Var f0 = fused(low, mid, high);
            Var g = f0;
            Var loss = null;
            for (int k = 0; k < unroll; k++) {
                g = predict(g, emb[k]);   // training-time test: later steps start from the head's own output
                Var ce = Functional.crossEntropy(g.matmul(head), want[k]);
                loss = loss == null ? ce : loss.add(ce);
                if (k == 0 && featureWeight > 0) {
                    loss = loss.add(g.sub(nextHigh).square().mean().mul(featureWeight));
                }
                {   // P-EAGLE: depth k from the fused feature and the first token, by its own MLP
                    Var p = predictParallel(parallelInput(f0.detach(), k, n), emb[0]);
                    loss = loss.add(Functional.crossEntropy(p.matmul(head), want[k]));
                }
            }
            loss.backward();
            optimizer.step(parameters());
            for (Var p : parameters()) {
                p.zeroGrad();
            }
            last = loss.value().item();
        }
        return last;
    }

    private static float[] flatten(List<float[]> rows) {
        int w = rows.getFirst().length;
        float[] out = new float[rows.size() * w];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, out, i * w, w);
        }
        return out;
    }

    /** A draft node: its token, its parent among the nodes (-1 for the root's children), its state. */
    private record Node(int token, int parent, float[] feature, double logProb) {
    }

    /** The fused feature of one row of the last target pass. */
    private float[] rootFeature(PagedLlama.Output o, int row, int rowsBefore) {
        int at = rowsBefore + row;
        Var low = Var.of(Tensor.of(Shape.of(1, hidden), Arrays.copyOfRange(target.tapped(0), at * hidden,
                (at + 1) * hidden)));
        Var mid = Var.of(Tensor.of(Shape.of(1, hidden), Arrays.copyOfRange(target.tapped(1), at * hidden,
                (at + 1) * hidden)));
        Var high = Var.of(Tensor.of(Shape.of(1, hidden), Arrays.copyOfRange(o.hidden(), row * hidden,
                (row + 1) * hidden)));
        return fused(low, mid, high).value().toArray();
    }

    /**
     * Greedy generation with a draft tree, drafted step by step (EAGLE-3).
     *
     * @param prompt the prompt
     * @param maxNew tokens to generate
     * @param topK children per expanded node
     * @param treeDepth the tree's depth
     * @param width the nodes expanded per depth
     * @return the tokens and counts
     */
    public Result generate(int[] prompt, int maxNew, int topK, int treeDepth, int width) {
        return run(prompt, maxNew, topK, treeDepth, width, false);
    }

    /**
     * Greedy generation with chains of {@code count} tokens drafted in one parallel pass
     * (P-EAGLE).
     *
     * @param prompt the prompt
     * @param maxNew tokens to generate
     * @param count the tokens drafted per pass, at most {@code unroll}
     * @return the tokens and counts
     */
    public Result generateParallel(int[] prompt, int maxNew, int count) {
        if (count < 1 || count > unroll) {
            throw new IllegalArgumentException("can draft 1 to " + unroll + " tokens in parallel");
        }
        return run(prompt, maxNew, 1, count, 1, true);
    }

    private Result run(int[] prompt, int maxNew, int topK, int treeDepth, int width, boolean parallel) {
        long t0 = System.nanoTime();
        KvBlockStore store = target.store();
        target.tap(taps);
        int vocab = target.vocabSize();
        int seq = store.newSequence();
        PagedLlama.Output first = target.forward(List.of(PagedLlama.Pass.chain(seq, prompt)), false).getFirst();
        int passes = 1;
        int pending = LogitMath.argmax(first.logits());
        float[] feature = rootFeature(first, 0, prompt.length - 1);
        List<Integer> out = new ArrayList<>();
        out.add(pending);
        int[] chain = {pending};
        int draftPasses = 0;
        int acceptedTotal = 0;
        Var head = Var.of(target.outputWeights());
        while (out.size() < maxNew) {
            List<Node> tree;
            if (parallel) {
                tree = parallelDraft(feature, pending, treeDepth, head);
                draftPasses++;
            } else {
                tree = draft(feature, pending, topK, treeDepth, width, head, vocab);
                draftPasses += treeDepth;
            }
            int c = chain.length;
            int[] tokens = new int[c + tree.size()];
            int[] parents = new int[tokens.length];
            for (int i = 0; i < c; i++) {
                tokens[i] = chain[i];
                parents[i] = i - 1;
            }
            for (int i = 0; i < tree.size(); i++) {
                tokens[c + i] = tree.get(i).token();
                parents[c + i] = tree.get(i).parent() < 0 ? c - 1 : c + tree.get(i).parent();
            }
            int before = store.length(seq);
            PagedLlama.Output o = target.forward(List.of(new PagedLlama.Pass(seq, tokens, parents)), true)
                    .getFirst();
            passes++;
            int row = c - 1;
            List<Integer> accepted = new ArrayList<>();
            int next = LogitMath.argmax(o.row(row, vocab));
            while (true) {
                int child = -1;
                for (int i = 0; i < tree.size(); i++) {
                    if (parents[c + i] == row && tree.get(i).token() == next) {
                        child = c + i;
                        break;
                    }
                }
                if (child < 0) {
                    break;
                }
                accepted.add(next);
                row = child;
                next = LogitMath.argmax(o.row(row, vocab));
            }
            acceptedTotal += accepted.size();
            out.addAll(accepted);
            out.add(next);
            feature = rootFeature(o, row, 0);
            pending = next;
            store.truncate(seq, before + c);
            chain = new int[accepted.size() + 1];
            for (int i = 0; i < accepted.size(); i++) {
                chain[i] = accepted.get(i);
            }
            chain[accepted.size()] = next;
        }
        store.free(seq);
        target.tap();
        int[] tokens = out.stream().mapToInt(Integer::intValue).limit(maxNew).toArray();
        return new Result(tokens, passes, draftPasses, acceptedTotal, (System.nanoTime() - t0) / 1e6);
    }

    /** P-EAGLE: every depth from the fused feature and the pending token, in one batched pass. */
    private List<Node> parallelDraft(float[] feature, int pending, int count, Var head) {
        Var f0 = Var.of(Tensor.of(Shape.of(1, hidden), feature)).expand(Shape.of(count, hidden));
        Var in = f0.add(depth.slice(0, 0, count));
        float[] e = target.embed(pending);
        float[] rows = new float[count * hidden];
        for (int k = 0; k < count; k++) {
            System.arraycopy(e, 0, rows, k * hidden, hidden);
        }
        float[] logits = predictParallel(in, Var.of(Tensor.of(Shape.of(count, hidden), rows))).matmul(head).value()
                .toArray();
        int vocab = target.vocabSize();
        List<Node> nodes = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            int best = LogitMath.argmax(Arrays.copyOfRange(logits, k * vocab, (k + 1) * vocab));
            nodes.add(new Node(best, k - 1, null, 0));
        }
        return nodes;
    }

    /** Expands the draft tree level by level, each level one draft pass over the frontier. */
    private List<Node> draft(float[] feature, int pending, int topK, int treeDepth, int width, Var head, int vocab) {
        List<Node> nodes = new ArrayList<>();
        List<Integer> frontier = new ArrayList<>();
        frontier.add(-1);
        for (int d = 0; d < treeDepth; d++) {
            int n = frontier.size();
            float[] f = new float[n * hidden];
            int[] toks = new int[n];
            for (int i = 0; i < n; i++) {
                int at = frontier.get(i);
                System.arraycopy(at < 0 ? feature : nodes.get(at).feature(), 0, f, i * hidden, hidden);
                toks[i] = at < 0 ? pending : nodes.get(at).token();
            }
            Var guess = predict(Var.of(Tensor.of(Shape.of(n, hidden), f)),
                    Var.of(Tensor.of(Shape.of(n, hidden), target.embed(toks))));
            float[] g = guess.value().toArray();
            float[] logits = guess.matmul(head).value().toArray();
            List<Integer> children = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int at = frontier.get(i);
                double base = at < 0 ? 0 : nodes.get(at).logProb();
                float[] row = Arrays.copyOfRange(logits, i * vocab, (i + 1) * vocab);
                double[] logp = LogitMath.logSoftmax(row);
                float[] feat = Arrays.copyOfRange(g, i * hidden, (i + 1) * hidden);
                for (int k = 0; k < topK; k++) {
                    int best = LogitMath.argmax(row);
                    row[best] = LogitMath.MASKED;
                    nodes.add(new Node(best, at, feat, base + logp[best]));
                    children.add(nodes.size() - 1);
                }
            }
            children.sort((a, b) -> Double.compare(nodes.get(b).logProb(), nodes.get(a).logProb()));
            frontier = new ArrayList<>(children.subList(0, Math.min(width, children.size())));
        }
        return nodes;
    }
}
