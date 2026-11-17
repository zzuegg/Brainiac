package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Weight-space algebra on models with the same architecture, each given as the list of
 * its parameter values (for example {@code LearningMath.values(model.parameters())}):
 *
 * <ul>
 *   <li><b>Task arithmetic</b> (Ilharco et al. 2023, <a href="https://arxiv.org/abs/2212.04089">arXiv:2212.04089</a>):
 *       a task vector is {@code finetuned - base}; adding vectors combines skills,
 *       subtracting one removes a skill.</li>
 *   <li><b>TIES-merging</b> (Yadav et al. 2023, <a href="https://arxiv.org/abs/2306.01708">arXiv:2306.01708</a>):
 *       trim each task vector to its largest entries, elect a sign per entry by total
 *       mass, and average only the entries that agree with it, which removes
 *       interference.</li>
 *   <li><b>DARE</b> (Yu et al. 2024, <a href="https://arxiv.org/abs/2311.03099">arXiv:2311.03099</a>):
 *       drop a random share {@code p} of a task vector's entries and rescale the rest by
 *       {@code 1/(1-p)}; delta weights are so redundant that even 90% can go.</li>
 *   <li><b>Model soups</b> (Wortsman et al. 2022, <a href="https://arxiv.org/abs/2203.05482">arXiv:2203.05482</a>):
 *       the uniform average of fine-tuned models, or the greedy soup that adds a model
 *       only if the held-out score does not drop.</li>
 * </ul>
 *
 * <p>Merging also works as a crossover operator for neuroevolution of trained networks:
 * two children of one ancestor combine what each learned
 * ({@link #crossover(List, List, List, double)}). {@link #ties(Model, List, double, float)}
 * merges whole models.
 */
public final class ModelMerging {

    private ModelMerging() {
    }

    /**
     * @param base the base weights
     * @param finetuned the fine-tuned weights
     * @return {@code finetuned - base}, per parameter
     */
    public static List<Tensor> taskVector(List<Tensor> base, List<Tensor> finetuned) {
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < base.size(); i++) {
            out.add(finetuned.get(i).sub(base.get(i)));
        }
        return out;
    }

    /**
     * @param base the base weights
     * @param vectors task vectors
     * @param scale the factor on their sum (negative to subtract a skill)
     * @return {@code base + scale * sum(vectors)}
     */
    public static List<Tensor> add(List<Tensor> base, List<List<Tensor>> vectors, float scale) {
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < base.size(); i++) {
            Tensor sum = null;
            for (List<Tensor> v : vectors) {
                sum = sum == null ? v.get(i) : sum.add(v.get(i));
            }
            out.add(base.get(i).add(sum.mul(scale)));
        }
        return out;
    }

    /**
     * TIES-merging of several fine-tuned models.
     *
     * @param base the base weights
     * @param finetuned the fine-tuned models
     * @param density the share of each task vector's entries kept by magnitude, for example 0.2
     * @param scale the factor on the merged vector, for example 1
     * @return the merged weights
     */
    public static List<Tensor> ties(List<Tensor> base, List<List<Tensor>> finetuned, double density, float scale) {
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < base.size(); i++) {
            int n = Math.toIntExact(base.get(i).size());
            float[][] trimmed = new float[finetuned.size()][];
            for (int k = 0; k < finetuned.size(); k++) {
                trimmed[k] = trim(finetuned.get(k).get(i).sub(base.get(i)).toArray(), density);
            }
            float[] merged = new float[n];
            for (int e = 0; e < n; e++) {
                double mass = 0;
                for (float[] t : trimmed) {
                    mass += t[e];
                }
                double sign = Math.signum(mass);
                double sum = 0;
                int count = 0;
                for (float[] t : trimmed) {
                    if (t[e] != 0 && Math.signum(t[e]) == sign) {
                        sum += t[e];
                        count++;
                    }
                }
                merged[e] = count == 0 ? 0 : (float) (sum / count);
            }
            out.add(base.get(i).add(Tensor.of(base.get(i).shape(), merged).mul(scale)));
        }
        return out;
    }

    /**
     * TIES-merging as a crossover: combines two descendants of a common ancestor.
     *
     * @param ancestor the common ancestor's weights
     * @param a one parent's weights
     * @param b the other parent's weights
     * @param density the share of each parent's changes kept
     * @return the child's weights
     */
    public static List<Tensor> crossover(List<Tensor> ancestor, List<Tensor> a, List<Tensor> b, double density) {
        return ties(ancestor, List.of(a, b), density, 1);
    }

    /**
     * TIES-merges whole models with the same architecture into a copy of the base.
     *
     * @param base the base model
     * @param finetuned models fine-tuned from it
     * @param density the share of each task vector's entries kept
     * @param scale the factor on the merged vector
     * @return a new model with the merged weights
     */
    public static Model ties(Model base, List<Model> finetuned, double density, float scale) {
        List<List<Tensor>> tuned = finetuned.stream().map(m -> LearningMath.values(m.parameters())).toList();
        List<Tensor> merged = ties(LearningMath.values(base.parameters()), tuned, density, scale);
        Model out = base.copy(base.device());
        List<Var> params = out.parameters();
        for (int i = 0; i < params.size(); i++) {
            Var p = params.get(i);
            p.assign(merged.get(i).to(p.value().device()).to(p.value().dtype()));
        }
        return out;
    }

    private static float[] trim(float[] v, double density) {
        int keep = (int) Math.ceil(density * v.length);
        float[] abs = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            abs[i] = Math.abs(v[i]);
        }
        float[] sorted = abs.clone();
        java.util.Arrays.sort(sorted);
        float threshold = keep == 0 ? Float.POSITIVE_INFINITY : sorted[v.length - keep];
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = abs[i] >= threshold ? v[i] : 0;
        }
        return out;
    }

    /**
     * DARE: drop and rescale a task vector.
     *
     * @param vector a task vector
     * @param dropRate the share of entries set to zero, for example 0.9
     * @param seed the seed
     * @return the sparsified vector, rescaled by {@code 1 / (1 - dropRate)}
     */
    public static List<Tensor> dare(List<Tensor> vector, double dropRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Tensor> out = new ArrayList<>();
        for (Tensor t : vector) {
            float[] v = t.toArray();
            for (int i = 0; i < v.length; i++) {
                v[i] = r.nextDouble() < dropRate ? 0 : (float) (v[i] / (1 - dropRate));
            }
            out.add(Tensor.of(t.shape(), v));
        }
        return out;
    }

    /**
     * @param models the models
     * @return the uniform average of their weights
     */
    public static List<Tensor> uniformSoup(List<List<Tensor>> models) {
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < models.get(0).size(); i++) {
            Tensor sum = null;
            for (List<Tensor> m : models) {
                sum = sum == null ? m.get(i) : sum.add(m.get(i));
            }
            out.add(sum.div(models.size()));
        }
        return out;
    }

    /**
     * The greedy soup: sort candidates by their own score, start from the best, and add
     * each next one to the average only if the score of the average does not drop.
     *
     * @param candidates the fine-tuned models
     * @param score a held-out score of weights, higher is better
     * @return the soup's weights
     */
    public static List<Tensor> greedySoup(List<List<Tensor>> candidates, ToDoubleFunction<List<Tensor>> score) {
        List<List<Tensor>> sorted = new ArrayList<>(candidates);
        sorted.sort((a, b) -> Double.compare(score.applyAsDouble(b), score.applyAsDouble(a)));
        List<List<Tensor>> ingredients = new ArrayList<>();
        ingredients.add(sorted.get(0));
        double best = score.applyAsDouble(sorted.get(0));
        for (int k = 1; k < sorted.size(); k++) {
            ingredients.add(sorted.get(k));
            double s = score.applyAsDouble(uniformSoup(ingredients));
            if (s >= best) {
                best = s;
            } else {
                ingredients.removeLast();
            }
        }
        return uniformSoup(ingredients);
    }
}
