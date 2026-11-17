package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;

/**
 * DuoAttention: only some heads need the whole context. <em>Retrieval heads</em> read far-back
 * tokens and keep a full KV cache; <em>streaming heads</em> attend to the first few tokens
 * (attention sinks) and a recent window, so they keep a constant-size cache. Which head is
 * which is found by optimisation, not by heuristics: every head gets a gate {@code a} that
 * mixes its full-attention and streaming-attention outputs, {@code a o_full + (1 - a)
 * o_stream}, and the gates are trained on calibration data to reproduce the full model's
 * outputs with an L1 penalty pushing them towards streaming. Heads whose gates stay high are
 * the retrieval heads.
 *
 * <p>{@link #gates} runs the optimisation on per-head outputs (optionally through each layer's
 * output projection, so heads interact as in the model); {@link #identify} keeps a share of
 * heads with the highest gates; the cache is then a {@link KvBudgetPolicy} whose selection
 * keeps every token for retrieval heads and sinks plus window for streaming heads (its memory
 * follows from that split, so the budget argument is an upper bound only).
 *
 * <p>Xiao et al., "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and
 * Streaming Heads" (ICLR 2025), <a href="https://arxiv.org/abs/2410.10819">arXiv:2410.10819</a>.
 */
public final class DuoHeadCache implements KvBudgetPolicy {

    private final boolean[][] retrieval;
    private final int sinks;
    private final int recent;

    /**
     * @param retrieval {@code [layer][head]} whether the head keeps the full context
     * @param sinks the first tokens a streaming head keeps
     * @param recent the most recent tokens a streaming head keeps
     */
    public DuoHeadCache(boolean[][] retrieval, int sinks, int recent) {
        if (sinks < 0 || recent < 1) {
            throw new IllegalArgumentException("need sinks >= 0 and recent >= 1");
        }
        this.retrieval = new boolean[retrieval.length][];
        for (int l = 0; l < retrieval.length; l++) {
            this.retrieval[l] = retrieval[l].clone();
        }
        this.sinks = sinks;
        this.recent = recent;
    }

    /**
     * Learns one gate per head.
     *
     * @param full per calibration sample, {@code [layer][head][dim]} outputs with full attention
     * @param streaming the same samples' outputs with sink-plus-window attention
     * @param projection per layer the output projection {@code [heads * dim, out]}, or
     *     {@code null} to compare head outputs directly
     * @param penalty the L1 weight on the gates
     * @param steps Adam steps
     * @return {@code [layer][head]} gates in (0, 1)
     */
    public static double[][] gates(List<float[][][]> full, List<float[][][]> streaming, Tensor[] projection,
            float penalty, int steps) {
        int samples = full.size();
        int layers = full.getFirst().length;
        int heads = full.getFirst()[0].length;
        int dim = full.getFirst()[0][0].length;
        double norm = 0;
        long count = 0;
        for (float[][][] sample : full) {
            for (float[][] layer : sample) {
                for (float[] head : layer) {
                    for (float v : head) {
                        norm += v * v;
                        count++;
                    }
                }
            }
        }
        float scale = (float) Math.sqrt(count / Math.max(norm, 1e-12));   // errors relative to a typical output
        Var[] logits = new Var[layers];
        Tensor[] gap = new Tensor[layers];
        for (int l = 0; l < layers; l++) {
            logits[l] = Var.param(Tensor.full(Shape.of(1, heads, 1), 2f));
            float[] b = new float[samples * heads * dim];
            for (int s = 0; s < samples; s++) {
                for (int h = 0; h < heads; h++) {
                    for (int c = 0; c < dim; c++) {
                        b[(s * heads + h) * dim + c] = (streaming.get(s)[l][h][c] - full.get(s)[l][h][c]) * scale;
                    }
                }
            }
            gap[l] = Tensor.of(Shape.of(samples, heads, dim), b);
        }
        Optimizer opt = Optimizer.adam(0.1f);
        List<Var> params = Arrays.asList(logits);
        for (int step = 0; step < steps; step++) {
            Var loss = null;
            for (int l = 0; l < layers; l++) {
                Var alpha = logits[l].sigmoid();
                // mixed - full = (1 - a) (stream - full)
                Var diff = alpha.neg().add(1f).mul(Var.of(gap[l]));
                if (projection != null) {
                    diff = diff.reshape(samples, heads * dim).matmul(Var.of(projection[l]));
                }
                Var term = diff.square().mean().add(alpha.mean().mul(penalty));
                loss = loss == null ? term : loss.add(term);
            }
            loss.backward();
            opt.step(params);
            for (Var p : params) {
                p.zeroGrad();
            }
        }
        double[][] out = new double[layers][heads];
        for (int l = 0; l < layers; l++) {
            float[] g = logits[l].sigmoid().value().toArray();
            for (int h = 0; h < heads; h++) {
                out[l][h] = g[h];
            }
        }
        return out;
    }

    /**
     * Keeps the heads with the highest gates as retrieval heads.
     *
     * @param gates {@code [layer][head]} learned gates
     * @param share the share of all heads that keep the full context
     * @param sinks the sink tokens of streaming heads
     * @param recent the window of streaming heads
     * @return the cache policy
     */
    public static DuoHeadCache identify(double[][] gates, double share, int sinks, int recent) {
        int layers = gates.length;
        int heads = gates[0].length;
        Integer[] order = new Integer[layers * heads];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(gates[b / heads][b % heads], gates[a / heads][a % heads]));
        boolean[][] retrieval = new boolean[layers][heads];
        int count = (int) Math.round(share * order.length);
        for (int i = 0; i < count; i++) {
            retrieval[order[i] / heads][order[i] % heads] = true;
        }
        return new DuoHeadCache(retrieval, sinks, recent);
    }

    /**
     * @param layer the layer
     * @param head the head
     * @return whether it keeps the full context
     */
    public boolean isRetrieval(int layer, int head) {
        return retrieval[layer][head];
    }

    /**
     * Which tokens one head keeps.
     *
     * @param layer the layer
     * @param head the head
     * @param length the cached tokens
     * @return per token whether it stays
     */
    public boolean[] mask(int layer, int head, int length) {
        boolean[] keep = new boolean[length];
        for (int t = 0; t < length; t++) {
            keep[t] = retrieval[layer][head] || t < sinks || t >= length - recent;
        }
        return keep;
    }

    @Override
    public boolean[][][] select(KvSnapshot snapshot, int budget) {
        boolean[][][] out = new boolean[snapshot.layers()][snapshot.heads()][];
        for (int l = 0; l < out.length; l++) {
            for (int h = 0; h < out[l].length; h++) {
                out[l][h] = mask(l, h, snapshot.length());
            }
        }
        return out;
    }

    @Override
    public String toString() {
        int r = 0;
        for (boolean[] layer : retrieval) {
            for (boolean b : layer) {
                r += b ? 1 : 0;
            }
        }
        return "DuoHeadCache(" + r + " retrieval heads, sinks " + sinks + ", window " + recent + ")";
    }
}
