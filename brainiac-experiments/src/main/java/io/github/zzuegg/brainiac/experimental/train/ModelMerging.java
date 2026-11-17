package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Linalg;
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
 *   <li><b>SLERP</b>, <b>Fisher-weighted averaging</b> and <b>RegMean</b>: see {@link #slerp},
 *       {@link #fisher} and {@link #regMean}.</li>
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

    /**
     * Spherical linear interpolation (SLERP, Shoemake 1985; the default two-model merge of
     * mergekit, Goddard et al. 2024, <a href="https://arxiv.org/abs/2403.13257">arXiv:2403.13257</a>),
     * per parameter tensor: {@code sin((1-t) w) / sin(w) a + sin(t w) / sin(w) b} with
     * {@code w} the angle between the flattened tensors. Unlike the linear average it keeps
     * the norm of the interpolated weights when the two directions differ, instead of
     * shrinking it by {@code cos(w / 2)}. Nearly parallel tensors are interpolated linearly.
     * Apply it to task vectors ({@link #taskVector}) and add the result to the base to merge
     * fine-tunes of one model.
     *
     * @param a the first model's tensors
     * @param b the second model's tensors
     * @param t the position, 0 gives {@code a} and 1 gives {@code b}
     * @return the interpolated tensors
     */
    public static List<Tensor> slerp(List<Tensor> a, List<Tensor> b, double t) {
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < a.size(); i++) {
            float[] x = a.get(i).toArray();
            float[] y = b.get(i).toArray();
            double nx = 0;
            double ny = 0;
            double dot = 0;
            for (int k = 0; k < x.length; k++) {
                nx += (double) x[k] * x[k];
                ny += (double) y[k] * y[k];
                dot += (double) x[k] * y[k];
            }
            double cos = nx == 0 || ny == 0 ? 1 : dot / Math.sqrt(nx * ny);
            double ca;
            double cb;
            if (Math.abs(cos) > 0.9995) {
                ca = 1 - t;
                cb = t;
            } else {
                double omega = Math.acos(Math.max(-1, Math.min(1, cos)));
                double s = Math.sin(omega);
                ca = Math.sin((1 - t) * omega) / s;
                cb = Math.sin(t * omega) / s;
            }
            float[] m = new float[x.length];
            for (int k = 0; k < m.length; k++) {
                m[k] = (float) (ca * x[k] + cb * y[k]);
            }
            out.add(Tensor.of(a.get(i).shape(), m));
        }
        return out;
    }

    /**
     * Fisher-weighted averaging (Matena and Raffel 2022,
     * <a href="https://arxiv.org/abs/2111.09832">arXiv:2111.09832</a>): each weight is the
     * average of the models' values weighted by their diagonal Fisher information,
     * {@code sum_i F_i theta_i / sum_i F_i}. A weight that one task's loss is sensitive to and
     * the other's is not takes that task's value, instead of being halved.
     *
     * @param models the models' tensors
     * @param fishers the diagonal Fisher of each model, shaped like its tensors (see
     *     {@link #diagonalFisher})
     * @param epsilon added to every Fisher value so weights no model cares about become the
     *     plain average
     * @return the merged tensors
     */
    public static List<Tensor> fisher(List<List<Tensor>> models, List<List<Tensor>> fishers, double epsilon) {
        List<Tensor> out = new ArrayList<>();
        for (int p = 0; p < models.get(0).size(); p++) {
            float[] num = new float[Math.toIntExact(models.get(0).get(p).shape().size())];
            float[] den = new float[num.length];
            for (int m = 0; m < models.size(); m++) {
                float[] w = models.get(m).get(p).toArray();
                float[] f = fishers.get(m).get(p).toArray();
                for (int k = 0; k < num.length; k++) {
                    double fk = f[k] + epsilon;
                    num[k] += (float) (fk * w[k]);
                    den[k] += (float) fk;
                }
            }
            for (int k = 0; k < num.length; k++) {
                num[k] /= den[k];
            }
            out.add(Tensor.of(models.get(0).get(p).shape(), num));
        }
        return out;
    }

    /**
     * The empirical diagonal Fisher: the mean of squared per-example gradients.
     *
     * @param perExampleGradients one list of gradient tensors per example
     * @return the mean squared gradient of each parameter
     */
    public static List<Tensor> diagonalFisher(List<List<Tensor>> perExampleGradients) {
        List<Tensor> out = new ArrayList<>();
        int n = perExampleGradients.size();
        for (int p = 0; p < perExampleGradients.get(0).size(); p++) {
            Tensor sum = null;
            for (List<Tensor> g : perExampleGradients) {
                Tensor sq = g.get(p).square();
                sum = sum == null ? sq : sum.add(sq);
            }
            out.add(sum.div(n));
        }
        return out;
    }

    /**
     * The Gram matrix {@code X^T X} of a linear layer's inputs, the statistic RegMean needs.
     *
     * @param inputs the layer's inputs on a task's data {@code [n, in]}
     * @return {@code [in, in]}
     */
    public static Tensor gram(Tensor inputs) {
        return inputs.transpose().matmul(inputs);
    }

    /**
     * RegMean (Jin et al. 2023, <a href="https://arxiv.org/abs/2212.09849">arXiv:2212.09849</a>):
     * closed-form merging of a linear layer {@code y = x W}. The merged weight minimizes the
     * summed squared difference to every model's outputs on that model's own inputs,
     * {@code W = (sum_i G_i)^-1 sum_i G_i W_i} with {@code G_i = X_i^T X_i}. The off-diagonal
     * entries of each Gram matrix are scaled by {@code alpha} (the paper's regularizer, which
     * pulls the solution towards a plain average and keeps the sum invertible).
     *
     * @param weights each model's weight {@code [in, out]}
     * @param grams each model's input Gram matrix {@code [in, in]} (see {@link #gram})
     * @param alpha the off-diagonal scale in {@code [0, 1]}, for example 0.9
     * @return the merged weight
     */
    public static Tensor regMean(List<Tensor> weights, List<Tensor> grams, double alpha) {
        int in = weights.get(0).shape().dim(0);
        Tensor lhs = null;
        Tensor rhs = null;
        for (int m = 0; m < weights.size(); m++) {
            float[] g = grams.get(m).toArray();
            for (int r = 0; r < in; r++) {
                for (int c = 0; c < in; c++) {
                    if (r != c) {
                        g[r * in + c] *= (float) alpha;
                    }
                }
            }
            Tensor gt = Tensor.of(grams.get(m).shape(), g);
            Tensor gw = gt.matmul(weights.get(m));
            lhs = lhs == null ? gt : lhs.add(gt);
            rhs = rhs == null ? gw : rhs.add(gw);
        }
        float[] l = lhs.toArray();
        double trace = 0;
        for (int r = 0; r < in; r++) {
            trace += l[r * in + r];
        }
        for (int r = 0; r < in; r++) {
            l[r * in + r] += (float) (1e-6 * trace / in + 1e-12);   // a whisper of ridge for unseen inputs
        }
        return Linalg.solve(Tensor.of(lhs.shape(), l), rhs);
    }
}
