package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Weight-sparse training with a hard L0 constraint: after every step of an inner optimizer,
 * each listed weight matrix keeps only its {@code k} largest-magnitude entries and the rest
 * are set to zero. The kept share anneals linearly from 1 to the target density over
 * {@code annealSteps}, so the network starts dense and is squeezed gradually. Unlike
 * {@link RigL}, nothing is regrown from gradients: whatever the inner optimizer pushes
 * above the cut comes back by itself, which is how the paper trains.
 *
 * <p>Trained this way, models express each task with far fewer weights, and the circuit
 * left after pruning to the task is small and disentangled. As an {@link Optimizer}
 * wrapper it drops into any training loop.
 *
 * <p>Gao et al. 2025, "Weight-sparse transformers have interpretable circuits",
 * <a href="https://arxiv.org/abs/2511.13653">arXiv:2511.13653</a>.
 */
public final class TopKWeightSparsity implements Optimizer {

    private final Optimizer inner;
    private final List<Var> weights;
    private final double density;
    private final long annealSteps;
    private final Map<Var, Tensor> masks = new IdentityHashMap<>();
    private long steps;

    /**
     * @param inner the optimizer that takes the steps
     * @param weights the matrices to keep sparse
     * @param density the final share of entries kept per matrix, in {@code (0, 1]}
     * @param annealSteps steps over which the kept share falls from 1 to {@code density}
     */
    public TopKWeightSparsity(Optimizer inner, List<Var> weights, double density, long annealSteps) {
        this.inner = inner;
        this.weights = List.copyOf(weights);
        this.density = density;
        this.annealSteps = Math.max(1, annealSteps);
    }

    /** @return the share kept at the current step */
    public double currentDensity() {
        double t = Math.min(1, steps / (double) annealSteps);
        return 1 - t * (1 - density);
    }

    @Override
    public void step(List<Var> parameters) {
        steps++;
        inner.step(parameters);
        double keep = currentDensity();
        for (Var w : weights) {
            float[] v = w.value().to(DType.FLOAT32).toArray();
            int k = Math.max(1, (int) Math.round(keep * v.length));
            float[] mags = new float[v.length];
            for (int i = 0; i < v.length; i++) {
                mags[i] = Math.abs(v[i]);
            }
            float[] sorted = mags.clone();
            Arrays.sort(sorted);
            float cut = sorted[v.length - k];
            float[] m = new float[v.length];
            int kept = 0;
            for (int i = 0; i < v.length; i++) {
                if (mags[i] > cut || (mags[i] == cut && kept < k)) {
                    m[i] = 1;
                    kept++;
                } else {
                    v[i] = 0;
                }
            }
            masks.put(w, Tensor.of(w.shape(), m));
            w.assign(Tensor.of(w.shape(), v, w.value().device()).to(w.dtype()));
        }
    }

    /**
     * @param w a constrained matrix
     * @return its current keep mask, or null before the first step
     */
    public Tensor mask(Var w) {
        return masks.get(w);
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

    @Override
    public String toString() {
        return "TopKWeightSparsity(" + inner + ", density " + density + ")";
    }
}
