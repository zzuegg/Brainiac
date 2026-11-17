package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Guards any optimizer against gradient and loss spikes, after SPAM (Huang et al. 2025,
 * <a href="https://arxiv.org/abs/2501.06842">arXiv:2501.06842</a>) and the loss-spike
 * skipping used in large-model training (PaLM, OLMo). Rare outlier batches (corrupt data,
 * a numerical blow-up) otherwise poison Adam's moments for thousands of steps: plain norm
 * clipping still feeds the outlier's direction into the momentum and its size into the
 * second moment, which then suppresses every later step.
 *
 * <ul>
 *   <li><b>Batch skipping.</b> When the global gradient norm exceeds {@code normFactor}
 *       times its running average (or {@link #observeLoss} saw a loss above
 *       {@code lossFactor} times its running average), the batch is dropped: gradients are
 *       cleared and the inner optimizer does not step.</li>
 *   <li><b>Spike-aware clipping</b> (SPAM): entries whose square exceeds {@code theta} times
 *       their running average {@code v} are clipped to {@code sign(g) sqrt(theta v)}, so a
 *       single bad coordinate cannot dominate.</li>
 *   <li><b>Momentum reset</b> (SPAM, optional): every {@code resetEvery} steps, and after
 *       every skipped batch, the inner optimizer's state is reset through the core
 *       {@link Optimizer#resetState} hook, followed by a linear learning-rate warm-up over
 *       {@code warmup} steps, since a reset Adam takes large first steps.</li>
 *   <li><b>Norm clipping</b> (optional): kept batches are clipped to a global norm after
 *       detection, so the guard replaces ordinary clipping.</li>
 * </ul>
 */
public final class SpikeGuard implements Optimizer {

    private final Optimizer inner;
    private final float normFactor;
    private final float lossFactor;
    private final float theta;
    private final int resetEvery;
    private final int warmup;
    private final float maxNorm;
    private final Map<Var, Tensor> second = new IdentityHashMap<>();
    private double normAverage;
    private double lossAverage;
    private int seen;
    private int lossesSeen;
    private boolean skipNext;
    private int skipped;
    private int resets;
    private int sinceReset = Integer.MAX_VALUE / 2;
    private float baseRate;

    /**
     * @param inner the optimizer to guard
     * @param normFactor skip when the gradient norm exceeds this multiple of its average, for
     *     example 5; {@code Float.POSITIVE_INFINITY} never skips on gradients
     * @param lossFactor skip when an observed loss exceeds this multiple of its average, for
     *     example 3; {@code Float.POSITIVE_INFINITY} never skips on losses
     * @param theta the per-entry spike threshold of SPAM, for example 5000; {@code 0} disables it
     * @param resetEvery steps between periodic momentum resets; {@code 0} resets only after skips,
     *     {@code -1} never
     * @param warmup steps of linear learning-rate warm-up after a reset
     * @param maxNorm the global norm the kept gradients are clipped to after spike detection
     *     (which sees the raw norm), so the guard replaces ordinary clipping; {@code 0} for none
     */
    public SpikeGuard(Optimizer inner, float normFactor, float lossFactor, float theta, int resetEvery,
            int warmup, float maxNorm) {
        if (!(normFactor > 1) || !(lossFactor > 1) || theta < 0 || warmup < 0 || maxNorm < 0) {
            throw new IllegalArgumentException("bad SpikeGuard settings");
        }
        this.inner = inner;
        this.normFactor = normFactor;
        this.lossFactor = lossFactor;
        this.theta = theta;
        this.resetEvery = resetEvery;
        this.warmup = warmup;
        this.maxNorm = maxNorm;
        this.baseRate = inner.learningRate();
    }

    /**
     * Batch skipping on gradient-norm spikes (5x the average) and SPAM clipping, no resets,
     * no norm clipping.
     *
     * @param inner the optimizer to guard
     * @return the guarded optimizer
     */
    public static SpikeGuard of(Optimizer inner) {
        return new SpikeGuard(inner, 5, Float.POSITIVE_INFINITY, 5000, -1, 0, 0);
    }

    /**
     * As {@link #of(Optimizer)}, with the kept batches clipped to a global norm: a drop-in
     * for norm clipping that also skips spikes.
     *
     * @param inner the optimizer to guard
     * @param maxNorm the global norm to clip to
     * @return the guarded optimizer
     */
    public static SpikeGuard of(Optimizer inner, float maxNorm) {
        return new SpikeGuard(inner, 5, Float.POSITIVE_INFINITY, 5000, -1, 0, maxNorm);
    }

    /**
     * Tells the guard this batch's loss before {@link #step}: if it is a spike, the next
     * step is skipped.
     *
     * @param loss the batch loss
     * @return whether it was a spike
     */
    public boolean observeLoss(float loss) {
        boolean spike = !Float.isFinite(loss) || lossesSeen >= 5 && loss > lossFactor * lossAverage;
        if (spike) {
            skipNext = true;
        } else {
            lossAverage = lossesSeen == 0 ? loss : 0.95 * lossAverage + 0.05 * loss;
            lossesSeen++;
        }
        return spike;
    }

    /**
     * The global L2 norm of the parameters' gradients.
     *
     * @param parameters the parameters
     * @return the norm, 0 if none has a gradient
     */
    public static float globalNorm(List<Var> parameters) {
        Tensor sum = null;
        for (Var p : parameters) {
            if (p.grad() != null) {
                Tensor s = p.grad().square().sum();
                sum = sum == null ? s : sum.add(s);
            }
        }
        return sum == null ? 0 : (float) Math.sqrt(sum.item());
    }

    /**
     * Scales all gradients down so that their global norm is at most {@code maxNorm}, the
     * usual norm clipping (the baseline this guard improves on).
     *
     * @param parameters the parameters
     * @param maxNorm the largest global norm
     * @return the norm before clipping
     */
    public static float clipGlobalNorm(List<Var> parameters, float maxNorm) {
        float norm = globalNorm(parameters);
        if (norm > maxNorm) {
            for (Var p : parameters) {
                if (p.grad() != null) {
                    p.setGrad(p.grad().mul(maxNorm / norm));
                }
            }
        }
        return norm;
    }

    @Override
    public void step(List<Var> parameters) {
        float norm = globalNorm(parameters);
        boolean spike = skipNext || !Float.isFinite(norm) || seen >= 5 && norm > normFactor * normAverage;
        skipNext = false;
        if (spike) {
            skipped++;
            for (Var p : parameters) {
                p.zeroGrad();
            }
            if (resetEvery >= 0) {
                reset(parameters);
            }
            return;
        }
        normAverage = seen == 0 ? norm : 0.9 * normAverage + 0.1 * norm;
        seen++;
        if (theta > 0) {
            for (Var p : parameters) {
                Tensor g = p.grad();
                if (g == null) {
                    continue;
                }
                Tensor v = second.get(p);
                if (v != null) {
                    Tensor limit = v.mul(theta).sqrt();
                    g = g.minimum(limit).maximum(limit.neg());
                    p.setGrad(g);
                }
                Tensor nv = v == null ? g.square() : v.mul(0.999f).add(g.square().mul(0.001f));
                Tensor.evaluate(List.of(nv, g));
                second.put(p, nv);
            }
        }
        if (maxNorm > 0) {
            clipGlobalNorm(parameters, maxNorm);
        }
        if (sinceReset < warmup) {
            inner.learningRate(baseRate * (sinceReset + 1) / (warmup + 1));
        } else if (sinceReset == warmup) {
            inner.learningRate(baseRate);
        }
        sinceReset++;
        inner.step(parameters);
        if (resetEvery > 0 && seen % resetEvery == 0) {
            reset(parameters);
        }
    }

    private void reset(List<Var> parameters) {
        for (Var p : parameters) {
            inner.resetState(p, Tensor.full(Shape.of(), 0f));
        }
        resets++;
        sinceReset = 0;
    }

    /** @return batches skipped so far */
    public int skipped() {
        return skipped;
    }

    /** @return momentum resets so far */
    public int resets() {
        return resets;
    }

    @Override
    public void resetState(Var parameter, Tensor keep) {
        inner.resetState(parameter, keep);
    }

    @Override
    public void forget(Var parameter) {
        second.remove(parameter);
        inner.forget(parameter);
    }

    @Override
    public float learningRate() {
        return baseRate;
    }

    @Override
    public void learningRate(float learningRate) {
        baseRate = learningRate;
        inner.learningRate(learningRate);
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        return inner.state(parameters);
    }

    @Override
    public void loadState(List<Var> parameters, Map<String, Tensor> state) {
        inner.loadState(parameters, state);
    }

    @Override
    public String toString() {
        return "SpikeGuard(" + inner + ")";
    }
}
