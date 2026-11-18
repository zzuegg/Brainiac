package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.experimental.train.ModelMerging;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.ToDoubleFunction;

/**
 * Evolutionary model merging in parameter space: CMA-ES searches the recipe that merges
 * several models fine-tuned from one base, scored by a small validation set, with no gradient
 * training. The recipe is a mixing coefficient per task vector ({@link ModelMerging#taskVector})
 * and per parameter tensor ("layer"), optionally after DARE sparsification
 * ({@link ModelMerging#dare}) of each task vector:
 * {@code theta_l = base_l + sum_k lambda_(k, l) tau_(k, l)}. With one coefficient per model
 * instead ({@link #perLayer} off) it searches the global task-arithmetic weights. The
 * data-flow half of the paper (choosing which layers of which model to stack) is not
 * implemented.
 *
 * <p>Akiba et al. 2025, "Evolutionary Optimization of Model Merging Recipes", Nature Machine
 * Intelligence, <a href="https://arxiv.org/abs/2403.13187">arXiv:2403.13187</a>.
 */
public final class EvolutionaryMerge {

    private final List<Tensor> base;
    private final List<List<Tensor>> vectors;
    private boolean perLayer = true;
    private long seed = 1;
    private double start = 0.5;
    private double[] best;
    private double bestFitness = Double.NEGATIVE_INFINITY;
    private final List<Double> history = new ArrayList<>();

    private EvolutionaryMerge(List<Tensor> base, List<List<Tensor>> vectors) {
        this.base = base;
        this.vectors = vectors;
    }

    /**
     * @param base the shared base model's weights
     * @param finetuned the fine-tuned models' weights
     * @return the search over their merges
     */
    public static EvolutionaryMerge of(List<Tensor> base, List<List<Tensor>> finetuned) {
        Objects.requireNonNull(base, "base");
        List<List<Tensor>> vectors = new ArrayList<>();
        for (List<Tensor> f : finetuned) {
            vectors.add(ModelMerging.taskVector(base, f));
        }
        return new EvolutionaryMerge(List.copyOf(base), vectors);
    }

    /**
     * Sparsifies every task vector with DARE before the search.
     *
     * @param dropRate the share of entries dropped
     * @param dareSeed the seed
     * @return this
     */
    public EvolutionaryMerge dare(double dropRate, long dareSeed) {
        for (int k = 0; k < vectors.size(); k++) {
            vectors.set(k, ModelMerging.dare(vectors.get(k), dropRate, dareSeed + k));
        }
        return this;
    }

    /**
     * @param on one coefficient per model and tensor (default) or per model
     * @return this
     */
    public EvolutionaryMerge perLayer(boolean on) {
        this.perLayer = on;
        return this;
    }

    /**
     * @param coefficient the coefficient every search starts from (default 0.5)
     * @return this
     */
    public EvolutionaryMerge startAt(double coefficient) {
        this.start = coefficient;
        return this;
    }

    /**
     * @param s the CMA-ES seed
     * @return this
     */
    public EvolutionaryMerge seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return the number of searched coefficients */
    public int dimension() {
        return vectors.size() * (perLayer ? base.size() : 1);
    }

    /**
     * @param coefficients the recipe ({@link #dimension} values, model-major)
     * @return the merged weights
     */
    public List<Tensor> merge(double[] coefficients) {
        List<Tensor> out = new ArrayList<>();
        for (int l = 0; l < base.size(); l++) {
            Tensor t = base.get(l);
            for (int k = 0; k < vectors.size(); k++) {
                double c = coefficients[perLayer ? k * base.size() + l : k];
                t = t.add(vectors.get(k).get(l).mul((float) c));
            }
            out.add(t);
        }
        return out;
    }

    /**
     * Searches the recipe that maximises a validation fitness.
     *
     * @param fitness the fitness of merged weights (higher is better)
     * @param evaluations the CMA-ES evaluation budget
     * @return the best merged weights
     */
    public List<Tensor> search(ToDoubleFunction<List<Tensor>> fitness, int evaluations) {
        double[] x0 = new double[dimension()];
        java.util.Arrays.fill(x0, start);
        CmaEs.of(x0, 0.3).seed(seed).minimize(c -> {
            double f = fitness.applyAsDouble(merge(c));
            if (f > bestFitness) {
                bestFitness = f;
                best = c.clone();
            }
            history.add(bestFitness);
            return -f;
        }, evaluations);
        return merge(best);
    }

    /** @return the best recipe found */
    public double[] best() {
        return best == null ? null : best.clone();
    }

    /** @return its validation fitness */
    public double bestFitness() {
        return bestFitness;
    }

    /** @return the best validation fitness after each evaluation */
    public List<Double> history() {
        return List.copyOf(history);
    }
}
