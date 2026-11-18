package io.github.zzuegg.brainiac.experimental.events;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * An intensity-free temporal point process (Shchur, Biloš and Günnemann, "Intensity-Free
 * Learning of Temporal Point Processes", ICLR 2020,
 * <a href="https://arxiv.org/abs/1909.12127">arXiv:1909.12127</a>). Instead of an intensity whose
 * integral needs Monte Carlo or a restrictive form, it models the density of the next inter-event
 * time directly as a mixture of log-normals whose weights, means and scales come from a GRU
 * encoding of the history:
 *
 * <pre>
 * p(tau | h) = sum_m w_m(h) LogNormal(tau; mu_m(h), sigma_m(h)),   p(k | h) = softmax(W h)
 * </pre>
 *
 * <p>The log-likelihood, the survival function of the last interval, the expected next time and
 * exact sampling (pick a component, draw a normal, exponentiate) are all closed-form, and the
 * mixture approximates any density on the positive half-line.
 */
public final class IntensityFreeTpp {

    private static final double LOG_2PI = Math.log(2 * Math.PI);

    private final int types;
    private final int components;
    private final Layer gru;
    private final Var head;
    private final Var headB;
    private final Var mark;
    private final Var markB;
    private final List<Var> params = new ArrayList<>();

    /**
     * Creates a model.
     *
     * @param types event types
     * @param hidden the GRU width
     * @param components log-normal mixture components
     * @param seed the initialisation seed
     */
    public IntensityFreeTpp(int types, int hidden, int components, long seed) {
        this.types = types;
        this.components = components;
        SplittableRandom r = new SplittableRandom(seed);
        gru = Recurrent.gru(hidden).returnSequences(true).create(Shape.of(1, types + 2), r);
        head = EventBatch.weight(r, hidden, 3 * components);
        headB = EventBatch.bias(3 * components);
        mark = EventBatch.weight(r, hidden, types);
        markB = EventBatch.bias(types);
        params.addAll(gru.parameters());
        params.addAll(List.of(head, headB, mark, markB));
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    /** Mixture parameters {@code [B, P, 3M]}: log-weights, means, log-scales. */
    private Var[] mixture(Var h) {
        Var g = h.matmul(head).add(headB);
        Var logW = EventBatch.logSoftmax(g.slice(-1, 0, components));
        Var mu = g.slice(-1, components, 2 * components);
        Var logSigma = g.slice(-1, 2 * components, 3 * components).clamp(-5f, 1.5f);
        return new Var[] {logW, mu, logSigma};
    }

    private Var perInterval(EventBatch b) {
        Var h = gru.forward(Var.of(b.inputs));
        Var[] mix = mixture(h);
        Var logTau = Var.of(b.gaps.log());
        Var z = logTau.sub(mix[1]).div(mix[2].exp());
        Var logDensity = mix[0].add(z.square().mul(-0.5f)).sub(mix[2]).sub((float) (0.5 * LOG_2PI))
                .logsumexp(-1, true).sub(logTau);
        Var survival = z.mul((float) (1 / Math.sqrt(2))).erf().neg().add(1f).mul(0.5f).maximum(1e-12f).log()
                .add(mix[0]).logsumexp(-1, true);
        Var logMark = EventBatch.logSoftmax(h.matmul(mark).add(markB)).mul(b.nextType).sum(-1, true);
        Var ends = Var.of(b.ends);
        return logDensity.add(logMark).mul(ends).add(survival.mul(ends.neg().add(1f)));
    }

    /**
     * @param batch sequences
     * @return the negative log-likelihood per event
     */
    public Var loss(List<EventSequence> batch) {
        EventBatch b = new EventBatch(batch, types);
        return b.negativeMeanPerEvent(perInterval(b));
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
            Var l = loss(mb);
            last = l.value().item();
            l.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * @param data held-out sequences
     * @return {@code {log-likelihood per event, next-type accuracy, next-time MAE of the mixture's median}}
     */
    public double[] evaluate(List<EventSequence> data) {
        return Var.noGrad(() -> {
            EventBatch b = new EventBatch(data, types);
            double ll = -b.negativeMeanPerEvent(perInterval(b)).value().item();
            Var h = gru.forward(Var.of(b.inputs));
            float[] scores = h.matmul(mark).add(markB).value().toArray();
            Var[] mix = mixture(h);
            float[] logW = mix[0].value().toArray();
            float[] mu = mix[1].value().toArray();
            float[] ls = mix[2].value().toArray();
            float[] next = b.nextType.toArray();
            float[] ends = b.ends.toArray();
            float[] gaps = b.gaps.toArray();
            int right = 0;
            int total = 0;
            double mae = 0;
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
                // the mixture's median by bisection on the log time (the mean of heavy log-normal
                // tails is unstable)
                double lo = -20;
                double hi = 20;
                for (int it = 0; it < 50; it++) {
                    double mid = (lo + hi) / 2;
                    double cdf = 0;
                    for (int m = 0; m < components; m++) {
                        int at = i * components + m;
                        cdf += Math.exp(logW[at]) * HawkesProcess.normalCdf((mid - mu[at]) / Math.exp(ls[at]));
                    }
                    if (cdf < 0.5) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                double median = Math.exp((lo + hi) / 2);
                mae += Math.abs(median - gaps[i]);
                total++;
            }
            return new double[] {ll, right / (double) Math.max(1, total), mae / Math.max(1, total)};
        });
    }

    /**
     * Samples a sequence exactly, one draw per event (re-encoding the prefix each step).
     *
     * @param horizon the window end
     * @param maxEvents a cap on the number of events (re-encoding makes each step cost the prefix)
     * @param random the source
     * @return the sequence
     */
    public EventSequence sample(double horizon, int maxEvents, RandomGenerator random) {
        return Var.noGrad(() -> {
            List<Double> times = new ArrayList<>();
            List<Integer> kinds = new ArrayList<>();
            double t = 0;
            while (true) {
                EventSequence prefix = new EventSequence(times.stream().mapToDouble(Double::doubleValue).toArray(),
                        kinds.stream().mapToInt(Integer::intValue).toArray(), horizon);
                EventBatch b = new EventBatch(List.of(prefix), types);
                Var h = gru.forward(Var.of(b.inputs)).slice(1, b.positions - 1, b.positions);
                Var[] mix = mixture(h);
                float[] logW = mix[0].value().toArray();
                float[] mu = mix[1].value().toArray();
                float[] ls = mix[2].value().toArray();
                double u = random.nextDouble();
                int m = 0;
                double acc = Math.exp(logW[0]);
                while (acc < u && m + 1 < components) {
                    acc += Math.exp(logW[++m]);
                }
                t += Math.exp(mu[m] + Math.exp(ls[m]) * random.nextGaussian());
                if (t >= horizon || times.size() >= maxEvents) {
                    break;
                }
                float[] logits = h.matmul(mark).add(markB).value().toArray();
                double max = Double.NEGATIVE_INFINITY;
                for (float l : logits) {
                    max = Math.max(max, l);
                }
                double[] p = new double[types];
                double sum = 0;
                for (int k = 0; k < types; k++) {
                    p[k] = Math.exp(logits[k] - max);
                    sum += p[k];
                }
                double v = random.nextDouble() * sum;
                int k = 0;
                double c = p[0];
                while (c < v && k + 1 < types) {
                    c += p[++k];
                }
                times.add(t);
                kinds.add(k);
            }
            return new EventSequence(times.stream().mapToDouble(Double::doubleValue).toArray(),
                    kinds.stream().mapToInt(Integer::intValue).toArray(), horizon);
        });
    }
}
