package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Iterative magnitude pruning with rewinding, the procedure behind the Lottery Ticket
 * Hypothesis (Frankle and Carbin 2019, <a href="https://arxiv.org/abs/1803.03635">arXiv:1803.03635</a>;
 * rewinding to an early iteration instead of the initialisation, Frankle et al. 2020,
 * <a href="https://arxiv.org/abs/1912.05671">arXiv:1912.05671</a>):
 *
 * <ol>
 *   <li>record the weights at the rewind point ({@link #snapshot()}, at initialisation or
 *       after a few hundred steps);</li>
 *   <li>train; {@link #prune(double)} the smallest-magnitude share of the surviving weights
 *       (globally across layers);</li>
 *   <li>{@link #rewind()} the survivors to their recorded values and train again; repeat.</li>
 * </ol>
 *
 * <p>The resulting sparse "winning ticket" trains to the dense network's accuracy.
 * {@link #masked(Optimizer)} keeps pruned weights at zero during training.
 */
public final class LotteryTicket {

    private final List<Var> weights;
    private final Map<Var, Tensor> masks = new IdentityHashMap<>();
    private final Map<Var, Tensor> rewindPoint = new IdentityHashMap<>();

    /**
     * @param weights the weights to prune (usually the matrices, not the biases)
     */
    public LotteryTicket(List<Var> weights) {
        this.weights = List.copyOf(weights);
        for (Var w : weights) {
            masks.put(w, Tensor.full(w.shape(), 1f));
        }
        snapshot();
    }

    /** Records the current weights as the rewind point. */
    public void snapshot() {
        for (Var w : weights) {
            rewindPoint.put(w, w.value());
        }
    }

    /**
     * Prunes the smallest-magnitude share of the surviving weights, across all layers.
     *
     * @param fraction the share of survivors to remove, for example 0.2
     */
    public void prune(double fraction) {
        List<Float> alive = new ArrayList<>();
        for (Var w : weights) {
            float[] v = w.value().toArray();
            float[] m = masks.get(w).toArray();
            for (int i = 0; i < v.length; i++) {
                if (m[i] != 0) {
                    alive.add(Math.abs(v[i]));
                }
            }
        }
        float[] sorted = new float[alive.size()];
        for (int i = 0; i < sorted.length; i++) {
            sorted[i] = alive.get(i);
        }
        Arrays.sort(sorted);
        int cut = (int) Math.floor(fraction * sorted.length);
        if (cut == 0) {
            return;
        }
        float threshold = sorted[cut - 1];
        int toRemove = cut;
        for (Var w : weights) {
            float[] v = w.value().toArray();
            float[] m = masks.get(w).toArray();
            for (int i = 0; i < v.length && toRemove > 0; i++) {
                if (m[i] != 0 && Math.abs(v[i]) <= threshold) {
                    m[i] = 0;
                    toRemove--;
                }
            }
            Tensor mask = Tensor.of(w.shape(), m);
            masks.put(w, mask);
            w.assign(w.value().mul(mask));
        }
    }

    /** Resets surviving weights to the rewind point and zeroes pruned ones. */
    public void rewind() {
        for (Var w : weights) {
            w.assign(rewindPoint.get(w).mul(masks.get(w)));
        }
    }

    /**
     * @param w a weight
     * @return its 0/1 mask
     */
    public Tensor mask(Var w) {
        return masks.get(w);
    }

    /** @return the share of pruned weights */
    public double sparsity() {
        double zero = 0;
        double total = 0;
        for (Var w : weights) {
            total += w.shape().size();
            zero += w.shape().size() - masks.get(w).sum().item();
        }
        return zero / total;
    }

    /**
     * @param inner the optimizer
     * @return an optimizer that masks gradients before {@code inner} steps and re-applies
     *     the masks after, so pruned weights stay zero
     */
    public Optimizer masked(Optimizer inner) {
        return new MaskedOptimizer(inner, masks);
    }

    /** Applies fixed masks around another optimizer. */
    static final class MaskedOptimizer implements Optimizer {
        private final Optimizer inner;
        private final Map<Var, Tensor> masks;

        MaskedOptimizer(Optimizer inner, Map<Var, Tensor> masks) {
            this.inner = inner;
            this.masks = masks;
        }

        @Override
        public void step(List<Var> parameters) {
            for (Var p : parameters) {
                Tensor m = masks.get(p);
                if (m != null && p.grad() != null) {
                    p.setGrad(p.grad().mul(m));
                }
            }
            inner.step(parameters);
            for (Var p : parameters) {
                Tensor m = masks.get(p);
                if (m != null) {
                    p.assign(p.value().mul(m));
                }
            }
        }

        @Override
        public void forget(Var parameter) {
            inner.forget(parameter);
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
    }
}
