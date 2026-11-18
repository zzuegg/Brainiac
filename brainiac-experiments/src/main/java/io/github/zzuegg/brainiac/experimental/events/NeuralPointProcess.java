package io.github.zzuegg.brainiac.experimental.events;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Neural temporal point processes: the event history is encoded by a network, and the next
 * event's conditional intensity is read off the encoding. Three intensity families:
 * <ul>
 *   <li>{@link Kind#RMTPP} (Du et al., KDD 2016,
 *       <a href="https://doi.org/10.1145/2939672.2939875">doi:10.1145/2939672.2939875</a>): a GRU
 *       over (type, log gap) and {@code lambda*(t) = exp(v.h_j + w (t - t_j) + b)}, whose
 *       compensator is closed-form, with marks from a softmax on {@code h_j};</li>
 *   <li>{@link Kind#CONTINUOUS_TIME}: the decaying cell of the neural Hawkes process (Mei and
 *       Eisner, NeurIPS 2017, <a href="https://arxiv.org/abs/1612.09328">arXiv:1612.09328</a>).
 *       After each event the GRU state sets a start value {@code c}, a target {@code c_bar} and
 *       decay rates {@code delta}; between events the cell drifts as
 *       {@code c_bar + (c - c_bar) exp(-delta s)}, and {@code lambda_k = softplus(w_k . tanh(cell))}
 *       can rise, fall or be inhibited. (Simplification: the drift is read off a discrete GRU
 *       rather than fed back into a continuous-time LSTM recurrence.)</li>
 *   <li>{@link Kind#TRANSFORMER}: the transformer Hawkes process (Zuo et al., ICML 2020,
 *       <a href="https://arxiv.org/abs/2002.09291">arXiv:2002.09291</a>): type embeddings plus a
 *       sinusoidal encoding of the event time, causal self-attention, and
 *       {@code lambda_k = softplus(alpha_k s + w_k . h_j + b_k)} between events.</li>
 * </ul>
 * The last two integrate their intensity by Monte Carlo (random points per interval in training,
 * stratified midpoints in evaluation). The likelihood includes the final interval up to the
 * horizon.
 */
public final class NeuralPointProcess {

    /** The intensity family. */
    public enum Kind {
        /** Recurrent marked TPP with an exponential-of-linear intensity. */
        RMTPP,
        /** Decaying continuous-time cell (neural Hawkes). */
        CONTINUOUS_TIME,
        /** Transformer Hawkes process. */
        TRANSFORMER
    }

    private final Kind kind;
    private final int types;
    private final int hidden;
    private final int samples;
    private final Layer gru;
    private final List<Var> params = new ArrayList<>();
    // RMTPP
    private Var v;
    private Var vb;
    private Var w;
    private Var mark;
    private Var markB;
    // continuous-time
    private Var cW;
    private Var cB;
    // transformer
    private Var embed;
    private Var wq;
    private Var wk;
    private Var wv;
    private Var ff1;
    private Var ff2;
    private Var alpha;
    // shared intensity head
    private Var out;
    private Var outB;

    /**
     * Creates a model.
     *
     * @param kind the intensity family
     * @param types event types
     * @param hidden the encoder width
     * @param samples Monte Carlo points per interval in training
     * @param seed the initialisation seed
     */
    public NeuralPointProcess(Kind kind, int types, int hidden, int samples, long seed) {
        this.kind = kind;
        this.types = types;
        this.hidden = hidden;
        this.samples = samples;
        SplittableRandom r = new SplittableRandom(seed);
        if (kind == Kind.TRANSFORMER) {
            gru = null;
            embed = EventBatch.weight(r, types + 1, hidden);
            wq = EventBatch.weight(r, hidden, hidden);
            wk = EventBatch.weight(r, hidden, hidden);
            wv = EventBatch.weight(r, hidden, hidden);
            ff1 = EventBatch.weight(r, hidden, 2 * hidden);
            ff2 = EventBatch.weight(r, 2 * hidden, hidden);
            alpha = Var.param(Tensor.full(Shape.of(types), -0.1f));
            out = EventBatch.weight(r, hidden, types);
            outB = EventBatch.bias(types);
            params.addAll(List.of(embed, wq, wk, wv, ff1, ff2, alpha, out, outB));
        } else {
            gru = Recurrent.gru(hidden).returnSequences(true).create(Shape.of(1, types + 2), r);
            params.addAll(gru.parameters());
            if (kind == Kind.RMTPP) {
                v = EventBatch.weight(r, hidden, 1);
                vb = EventBatch.bias(1);
                w = Var.param(Tensor.full(Shape.of(1), 0.1f));
                mark = EventBatch.weight(r, hidden, types);
                markB = EventBatch.bias(types);
                params.addAll(List.of(v, vb, w, mark, markB));
            } else {
                cW = EventBatch.weight(r, hidden, 3 * hidden);
                cB = EventBatch.bias(3 * hidden);
                out = EventBatch.weight(r, hidden, types);
                outB = EventBatch.bias(types);
                params.addAll(List.of(cW, cB, out, outB));
            }
        }
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    /** @return the intensity family */
    public Kind kind() {
        return kind;
    }

    private Var encode(EventBatch b) {
        if (kind != Kind.TRANSFORMER) {
            return gru.forward(Var.of(b.inputs));
        }
        int bs = b.batch;
        int p = b.positions;
        float[] ids = b.typeIds.toArray();
        float[] onehot = new float[bs * p * (types + 1)];
        for (int i = 0; i < ids.length; i++) {
            onehot[i * (types + 1) + (int) ids[i]] = 1;
        }
        Var x = Var.of(Tensor.of(Shape.of(bs, p, types + 1), onehot)).matmul(embed)
                .add(Var.of(timeEncoding(b.times, hidden)));
        int heads = 2;
        int d = hidden / heads;
        Var q = x.matmul(wq).reshape(bs, p, heads, d).permute(0, 2, 1, 3);
        Var k = x.matmul(wk).reshape(bs, p, heads, d).permute(0, 2, 1, 3);
        Var vv = x.matmul(wv).reshape(bs, p, heads, d).permute(0, 2, 1, 3);
        Var att = Functional.attention(q, k, vv, true, (float) (1 / Math.sqrt(d))).permute(0, 2, 1, 3)
                .reshape(bs, p, hidden);
        Var h = x.add(att);
        return h.add(h.matmul(ff1).relu().matmul(ff2));
    }

    /** Sinusoidal encoding of absolute times {@code [B, P, 1] -> [B, P, dim]}. */
    static Tensor timeEncoding(Tensor times, int dim) {
        float[] t = times.toArray();
        float[] out = new float[t.length * dim];
        for (int i = 0; i < t.length; i++) {
            for (int j = 0; j < dim / 2; j++) {
                double f = Math.pow(100, -2.0 * j / dim);
                out[i * dim + 2 * j] = (float) Math.sin(t[i] * f);
                out[i * dim + 2 * j + 1] = (float) Math.cos(t[i] * f);
            }
        }
        Shape s = times.shape();
        return Tensor.of(Shape.of(s.dim(0), s.dim(1), dim), out);
    }

    /** Intensities {@code [B, P, M, K]} at lags {@code [B, P, M, 1]} after each position. */
    private Var intensityAt(Var h, Tensor lags) {
        int bs = h.shape().dim(0);
        int p = h.shape().dim(1);
        if (kind == Kind.CONTINUOUS_TIME) {
            Var g = h.matmul(cW).add(cB);
            Var c = g.slice(-1, 0, hidden).tanh().reshape(bs, p, 1, hidden);
            Var cbar = g.slice(-1, hidden, 2 * hidden).tanh().reshape(bs, p, 1, hidden);
            Var delta = g.slice(-1, 2 * hidden, 3 * hidden).softplus().reshape(bs, p, 1, hidden);
            Var cell = cbar.add(c.sub(cbar).mul(delta.mul(Var.of(lags)).neg().exp()));
            return cell.tanh().matmul(out).add(outB).softplus();
        }
        Var base = h.matmul(out).add(outB).reshape(bs, p, 1, types);
        return base.add(Var.of(lags).mul(alpha)).softplus();
    }

    private Var perInterval(EventBatch b, Tensor lags) {
        Var h = encode(b);
        int bs = b.batch;
        int p = b.positions;
        if (kind == Kind.RMTPP) {
            Var a = h.matmul(v).add(vb);
            Var gap = Var.of(b.gaps);
            Var wd = gap.mul(w);
            Var integral = a.exp().mul(wd.exp().sub(1f)).div(w);
            Var logMark = EventBatch.logSoftmax(h.matmul(mark).add(markB)).mul(b.nextType).sum(-1, true);
            return a.add(wd).add(logMark).mul(b.ends).sub(integral);
        }
        Var lam = intensityAt(h, lags);
        Var integral = lam.sum(-1, false).mean(-1, false).reshape(bs, p, 1).mul(Var.of(b.gaps));
        Var atEvent = intensityAt(h, b.gaps.reshape(Shape.of(bs, p, 1, 1))).reshape(bs, p, types);
        Var logLam = atEvent.add(1e-8f).log().mul(b.nextType).sum(-1, true);
        return logLam.mul(b.ends).sub(integral);
    }

    /**
     * @param batch training sequences
     * @param random draws the Monte Carlo points
     * @return the negative log-likelihood per event
     */
    public Var loss(List<EventSequence> batch, RandomGenerator random) {
        EventBatch b = new EventBatch(batch, types);
        return b.negativeMeanPerEvent(perInterval(b, kind == Kind.RMTPP ? null : b.lags(samples, random)));
    }

    /**
     * Trains with Adam on random mini-batches.
     *
     * @param data training sequences
     * @param steps optimiser steps
     * @param batch sequences per step
     * @param lr the learning rate
     * @param seed the batch sampling seed
     * @return the final training loss
     */
    public double train(List<EventSequence> data, int steps, int batch, float lr, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        double last = 0;
        for (int s = 0; s < steps; s++) {
            List<EventSequence> mb = new ArrayList<>();
            for (int i = 0; i < batch; i++) {
                mb.add(data.get(r.nextInt(data.size())));
            }
            Var l = loss(mb, r);
            last = l.value().item();
            l.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * Evaluates with 32 stratified Monte Carlo points per interval.
     *
     * @param data held-out sequences
     * @return {@code {log-likelihood per event, next-type accuracy given the time}}
     */
    public double[] evaluate(List<EventSequence> data) {
        return Var.noGrad(() -> {
            EventBatch b = new EventBatch(data, types);
            double ll = -b.negativeMeanPerEvent(perInterval(b, kind == Kind.RMTPP ? null : b.lags(32, null)))
                    .value().item();
            Var h = encode(b);
            float[] scores = kind == Kind.RMTPP ? h.matmul(mark).add(markB).value().toArray()
                    : intensityAt(h, b.gaps.reshape(Shape.of(b.batch, b.positions, 1, 1))).value().toArray();
            float[] next = b.nextType.toArray();
            float[] ends = b.ends.toArray();
            int right = 0;
            int total = 0;
            for (int i = 0; i < ends.length; i++) {
                if (ends[i] == 0) {
                    continue;
                }
                int best = 0;
                for (int k = 1; k < types; k++) {
                    if (scores[i * types + k] > scores[i * types + best]) {
                        best = k;
                    }
                }
                right += next[i * types + best] > 0 ? 1 : 0;
                total++;
            }
            return new double[] {ll, right / (double) Math.max(1, total)};
        });
    }

    @Override
    public String toString() {
        return "NeuralPointProcess(" + kind + ", " + types + " types, hidden " + hidden + ")";
    }
}
