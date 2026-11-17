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
 * EAGLE speculative decoding: a small draft head that works on the target model's own
 * features instead of on tokens. From the target's last hidden state {@code f_t} and the
 * embedding of the next token {@code x_(t+1)} it predicts the next hidden state
 * {@code f_(t+1)}, which the target's own output layer turns into draft tokens; feeding its
 * prediction back drafts further. Because the feature already holds what the target has
 * understood, a two-layer head drafts far better than a separate small model of the same size.
 *
 * <p>Drafts form a tree (EAGLE-2): at each depth the {@code width} most likely nodes (by
 * cumulative draft probability) each propose their {@code topK} tokens. The target checks the
 * whole tree in one {@link PagedLlama} pass with tree attention, each node seeing only its
 * ancestors; the longest path that matches the target's greedy choices is kept, plus one token
 * of the target's own. Output equals greedy decoding of the target exactly; what changes is
 * the number of target passes. Kept nodes are read again as a plain chain at the start of the
 * next pass, so the cache never needs compacting.
 *
 * <p>Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (ICML
 * 2024), <a href="https://arxiv.org/abs/2401.15077">arXiv:2401.15077</a>; "EAGLE-2: Faster
 * Inference of Language Models with Dynamic Draft Trees" (2024),
 * <a href="https://arxiv.org/abs/2406.16858">arXiv:2406.16858</a>.
 */
public final class EagleDraft {

    /**
     * A generation.
     *
     * @param tokens the new tokens, equal to the target's greedy decoding
     * @param passes the target passes run
     * @param drafted the draft nodes the target checked
     * @param accepted the draft tokens kept
     */
    public record Result(int[] tokens, int passes, int drafted, int accepted) {

        /** @return new tokens per target pass */
        public double tokensPerPass() {
            return tokens.length / (double) passes;
        }
    }

    private final PagedLlama target;
    private final int hidden;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;

    /**
     * @param target the target model; its weights stay frozen
     * @param seed seeds the head's weights
     */
    public EagleDraft(PagedLlama target, long seed) {
        this.target = target;
        this.hidden = target.hiddenSize();
        SplittableRandom r = new SplittableRandom(seed);
        int in = 2 * hidden;
        w1 = Var.param(Tensor.normal(Shape.of(in, in), 0, (float) Math.sqrt(1.0 / in), r));
        b1 = Var.param(Tensor.zeros(Shape.of(in)));
        w2 = Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(0.5 / in), r));
        b2 = Var.param(Tensor.zeros(Shape.of(hidden)));
    }

    /** @return the head's trainable weights */
    public List<Var> parameters() {
        return List.of(w1, b1, w2, b2);
    }

    /** {@code f + MLP([f, e])}: the next feature from rows of features and token embeddings. */
    private Var predict(Var features, Var embeddings) {
        Var x = Var.concat(1, features, embeddings);
        Var h = x.matmul(w1).add(b1).activation(ActivationOp.SILU, 0);
        return features.add(h.matmul(w2).add(b2));
    }

    /**
     * Trains the head on the target's own features over sequences (for example the target's
     * greedy continuations of prompts, as EAGLE trains on the target's outputs): a feature
     * regression loss plus the cross-entropy of the target's output layer on the predicted
     * feature, with the true next-next token.
     *
     * @param sequences token sequences, at least three tokens each
     * @param steps optimizer steps, each over every position
     * @param optimizer the optimizer
     * @return the last loss
     */
    public float train(int[][] sequences, int steps, Optimizer optimizer) {
        KvBlockStore store = target.store();
        List<float[]> f = new ArrayList<>();
        List<float[]> e = new ArrayList<>();
        List<float[]> next = new ArrayList<>();
        List<Integer> follow = new ArrayList<>();
        for (int[] s : sequences) {
            int seq = store.newSequence();
            PagedLlama.Output out = target.forward(List.of(PagedLlama.Pass.chain(seq, s)), true).getFirst();
            store.free(seq);
            float[] emb = target.embed(s);
            for (int t = 0; t + 2 < s.length; t++) {
                f.add(Arrays.copyOfRange(out.hidden(), t * hidden, (t + 1) * hidden));
                e.add(Arrays.copyOfRange(emb, (t + 1) * hidden, (t + 2) * hidden));
                next.add(Arrays.copyOfRange(out.hidden(), (t + 1) * hidden, (t + 2) * hidden));
                follow.add(s[t + 2]);
            }
        }
        int n = f.size();
        Var features = Var.of(Tensor.of(Shape.of(n, hidden), flatten(f)));
        Var embeds = Var.of(Tensor.of(Shape.of(n, hidden), flatten(e)));
        Var wanted = Var.of(Tensor.of(Shape.of(n, hidden), flatten(next)));
        float[] ids = new float[n];
        for (int i = 0; i < n; i++) {
            ids[i] = follow.get(i);
        }
        Tensor labels = Tensor.of(Shape.of(n), ids);
        Var head = Var.of(target.outputWeights());
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            Var guess = predict(features, embeds);
            Var loss = guess.sub(wanted).square().mean().add(Functional.crossEntropy(guess.matmul(head), labels)
                    .mul(0.1f));
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

    /** A draft node: its token, its parent among the nodes (-1 for the root's children), its feature. */
    private record Node(int token, int parent, float[] feature, double logProb) {
    }

    /**
     * Generates greedily with tree drafts.
     *
     * @param prompt the prompt
     * @param maxNew tokens to generate
     * @param topK children per expanded node
     * @param depth the tree's depth
     * @param width the nodes expanded per depth
     * @return the tokens and counts
     */
    public Result generate(int[] prompt, int maxNew, int topK, int depth, int width) {
        KvBlockStore store = target.store();
        int vocab = target.vocabSize();
        int seq = store.newSequence();
        PagedLlama.Output first = target.forward(List.of(PagedLlama.Pass.chain(seq, prompt)), false).getFirst();
        int passes = 1;
        int pending = LogitMath.argmax(first.logits());
        float[] feature = first.hidden();   // the target's feature at the token before `pending`
        List<Integer> out = new ArrayList<>();
        out.add(pending);
        int[] chain = {pending};
        int drafted = 0;
        int acceptedTotal = 0;
        Var head = Var.of(target.outputWeights());
        while (out.size() < maxNew) {
            List<Node> tree = draft(feature, pending, topK, depth, width, head, vocab);
            drafted += tree.size();
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
            feature = Arrays.copyOfRange(o.hidden(), row * hidden, (row + 1) * hidden);
            pending = next;
            store.truncate(seq, before + c);   // keep the chain; kept tree nodes are read again next
            chain = new int[accepted.size() + 1];
            for (int i = 0; i < accepted.size(); i++) {
                chain[i] = accepted.get(i);
            }
            chain[accepted.size()] = next;
        }
        store.free(seq);
        int[] tokens = out.stream().mapToInt(Integer::intValue).limit(maxNew).toArray();
        return new Result(tokens, passes, drafted, acceptedTotal);
    }

    /** Expands the draft tree level by level from the feature before the pending token. */
    private List<Node> draft(float[] feature, int pending, int topK, int depth, int width, Var head, int vocab) {
        List<Node> nodes = new ArrayList<>();
        List<Integer> frontier = new ArrayList<>();   // node indices; -1 is the pending token
        frontier.add(-1);
        for (int d = 0; d < depth; d++) {
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
