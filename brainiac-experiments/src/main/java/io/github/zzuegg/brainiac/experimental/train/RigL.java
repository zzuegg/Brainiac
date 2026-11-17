package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * RigL, "Rigging the Lottery" (Evci et al. 2020, <a href="https://arxiv.org/abs/1911.11134">arXiv:1911.11134</a>):
 * dynamic sparse training for any weight matrices, as an {@link Optimizer} wrapper. Weights
 * start sparse at a fixed density and stay that sparse; every {@code interval} steps, in
 * each matrix, the share {@code alpha(t)} of active connections with the smallest
 * magnitude is dropped and as many inactive ones with the largest <em>dense</em> gradient
 * are grown (at zero). {@code alpha} anneals to zero with a cosine schedule by
 * {@code endStep}. Memory and compute stay sparse-sized throughout, yet the topology
 * keeps improving. (The core's {@code Plasticity} does the same for {@code Brain}
 * projections; this works on plain {@code Dense} weights.)
 *
 * <p>The forward pass must use the weight itself (not weight times mask), so that the
 * gradient of missing connections is available for growing.
 */
public final class RigL implements Optimizer {

    private final Optimizer inner;
    private final Map<Var, Tensor> masks = new IdentityHashMap<>();
    private final List<Var> weights;
    private final double alpha;
    private final int interval;
    private final long endStep;
    private long steps;
    private int updates;

    /**
     * @param inner the optimizer for the active weights
     * @param weights the matrices to keep sparse
     * @param density the share of connections kept, in {@code (0, 1]}
     * @param alpha the initial drop share, for example 0.3
     * @param interval steps between topology updates, for example 100
     * @param endStep the step after which the topology is fixed
     * @param seed the seed of the initial random masks
     */
    public RigL(Optimizer inner, List<Var> weights, double density, double alpha, int interval, long endStep,
            long seed) {
        this.inner = inner;
        this.weights = List.copyOf(weights);
        this.alpha = alpha;
        this.interval = interval;
        this.endStep = endStep;
        SplittableRandom r = new SplittableRandom(seed);
        for (Var w : weights) {
            int n = Math.toIntExact(w.shape().size());
            int keep = Math.max(1, (int) Math.round(density * n));
            float[] m = new float[n];
            int[] idx = r.ints(0, n).distinct().limit(keep).toArray();
            for (int i : idx) {
                m[i] = 1;
            }
            Tensor mask = Tensor.of(w.shape(), m);
            masks.put(w, mask);
            w.assign(w.value().mul(mask));
        }
    }

    /** @return how many topology updates have happened */
    public int updates() {
        return updates;
    }

    /**
     * @param w a weight
     * @return its current 0/1 mask
     */
    public Tensor mask(Var w) {
        return masks.get(w);
    }

    /** @return the share of active connections over all managed weights */
    public double density() {
        double on = 0;
        double total = 0;
        for (Var w : weights) {
            on += masks.get(w).sum().item();
            total += w.shape().size();
        }
        return on / total;
    }

    @Override
    public void step(List<Var> parameters) {
        steps++;
        if (steps % interval == 0 && steps < endStep) {
            double frac = alpha / 2 * (1 + Math.cos(Math.PI * steps / endStep));
            for (Var w : weights) {
                if (w.grad() != null) {
                    rewire(w, frac);
                }
            }
            updates++;
        }
        for (Var w : weights) {
            if (w.grad() != null) {
                w.setGrad(w.grad().mul(masks.get(w)));
            }
        }
        inner.step(parameters);
        for (Var w : weights) {
            w.assign(w.value().mul(masks.get(w)));
        }
    }

    @Override
    public void forget(Var parameter) {
        inner.forget(parameter);
    }

    private void rewire(Var w, double frac) {
        float[] v = w.value().toArray();
        float[] g = w.grad().toArray();
        float[] m = masks.get(w).toArray();
        List<Integer> active = new ArrayList<>();
        List<Integer> inactive = new ArrayList<>();
        for (int i = 0; i < m.length; i++) {
            (m[i] != 0 ? active : inactive).add(i);
        }
        int k = Math.min((int) Math.floor(frac * active.size()), inactive.size());
        if (k == 0) {
            return;
        }
        Integer[] drop = active.toArray(Integer[]::new);
        Arrays.sort(drop, (a, b) -> Float.compare(Math.abs(v[a]), Math.abs(v[b])));
        Integer[] grow = inactive.toArray(Integer[]::new);
        Arrays.sort(grow, (a, b) -> Float.compare(Math.abs(g[b]), Math.abs(g[a])));
        for (int i = 0; i < k; i++) {
            m[drop[i]] = 0;
            v[drop[i]] = 0;
        }
        for (int i = 0; i < k; i++) {
            m[grow[i]] = 1;
            v[grow[i]] = 0;   // grown connections start at zero
        }
        masks.put(w, Tensor.of(w.shape(), m));
        w.assign(Tensor.of(w.shape(), v));
    }

    @Override
    public float learningRate() {
        return inner.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
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
        return "RigL(" + inner + ")";
    }
}
