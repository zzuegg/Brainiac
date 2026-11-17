package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.nn.Model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Merges models fine-tuned from a common base into one, by combining their <em>task
 * vectors</em> ({@code finetuned - base}). Besides plain averaging and task arithmetic it
 * implements TIES-merging, which removes the interference that makes naive averaging fail:
 *
 * <ol>
 *   <li><b>Trim</b>: keep only the largest {@code density} share of each task vector's
 *       entries by magnitude (the rest is mostly fine-tuning noise);</li>
 *   <li><b>Elect sign</b>: per parameter, the sign with the larger total magnitude
 *       wins;</li>
 *   <li><b>Disjoint merge</b>: average only the entries that agree with the elected sign,
 *       so a parameter one task needs is not halved by tasks that left it alone.</li>
 * </ol>
 *
 * <p>The same operation is a crossover operator for neuroevolution of trained networks:
 * two children of one ancestor combine what each learned
 * ({@link #crossover(double[], double[], double[], double)}).
 *
 * <p>Yadav et al. 2023, "TIES-Merging: Resolving Interference When Merging Models",
 * <a href="https://arxiv.org/abs/2306.01708">arXiv:2306.01708</a>; Ilharco et al. 2023,
 * "Editing Models with Task Arithmetic", <a href="https://arxiv.org/abs/2212.04089">arXiv:2212.04089</a>.
 */
public final class ModelMerging {

    private ModelMerging() {
    }

    /**
     * The difference {@code finetuned - base}.
     *
     * @param base the base parameters
     * @param finetuned the fine-tuned parameters
     * @return the task vector
     */
    public static double[] taskVector(double[] base, double[] finetuned) {
        check(base, finetuned);
        double[] out = new double[base.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = finetuned[i] - base[i];
        }
        return out;
    }

    /**
     * The element-wise mean of several parameter vectors.
     *
     * @param models the parameter vectors
     * @return their mean
     */
    public static double[] average(List<double[]> models) {
        double[] out = new double[models.getFirst().length];
        for (double[] m : models) {
            check(out, m);
            for (int i = 0; i < out.length; i++) {
                out[i] += m[i] / models.size();
            }
        }
        return out;
    }

    /**
     * Task arithmetic: {@code base + lambda * sum(task vectors)}.
     *
     * @param base the base parameters
     * @param finetuned the fine-tuned parameter vectors
     * @param lambda the scale
     * @return the merged parameters
     */
    public static double[] taskArithmetic(double[] base, List<double[]> finetuned, double lambda) {
        double[] out = base.clone();
        for (double[] f : finetuned) {
            double[] t = taskVector(base, f);
            for (int i = 0; i < out.length; i++) {
                out[i] += lambda * t[i];
            }
        }
        return out;
    }

    /**
     * Keeps the largest {@code density} share of entries by magnitude, zeroing the rest.
     *
     * @param v the vector
     * @param density the share to keep, in {@code (0, 1]}
     * @return the trimmed copy
     */
    public static double[] trim(double[] v, double density) {
        if (!(density > 0 && density <= 1)) {
            throw new IllegalArgumentException("density must be in (0, 1]");
        }
        int keep = (int) Math.ceil(density * v.length);
        double[] mags = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            mags[i] = Math.abs(v[i]);
        }
        double[] sorted = mags.clone();
        Arrays.sort(sorted);
        double cut = sorted[v.length - keep];
        double[] out = new double[v.length];
        int kept = 0;
        for (int i = 0; i < v.length && kept < keep; i++) {
            if (mags[i] >= cut && mags[i] > 0) {
                out[i] = v[i];
                kept++;
            }
        }
        return out;
    }

    /**
     * TIES-merging.
     *
     * @param base the base parameters
     * @param finetuned the fine-tuned parameter vectors
     * @param density the share of each task vector kept by trimming
     * @param lambda the scale of the merged task vector
     * @return the merged parameters
     */
    public static double[] ties(double[] base, List<double[]> finetuned, double density, double lambda) {
        List<double[]> tasks = new ArrayList<>();
        for (double[] f : finetuned) {
            tasks.add(trim(taskVector(base, f), density));
        }
        double[] out = base.clone();
        for (int i = 0; i < out.length; i++) {
            double total = 0;
            for (double[] t : tasks) {
                total += t[i];
            }
            double sign = Math.signum(total);
            if (sign == 0) {
                continue;
            }
            double sum = 0;
            int count = 0;
            for (double[] t : tasks) {
                if (t[i] != 0 && Math.signum(t[i]) == sign) {
                    sum += t[i];
                    count++;
                }
            }
            if (count > 0) {
                out[i] += lambda * sum / count;
            }
        }
        return out;
    }

    /**
     * TIES-merging as a crossover: combines two descendants of a common ancestor.
     *
     * @param ancestor the common ancestor's parameters
     * @param a one parent's parameters
     * @param b the other parent's parameters
     * @param density the share of each parent's changes kept
     * @return the child's parameters
     */
    public static double[] crossover(double[] ancestor, double[] a, double[] b, double density) {
        return ties(ancestor, List.of(a, b), density, 1);
    }

    /**
     * TIES-merges whole models with the same architecture into a copy of the base.
     *
     * @param base the base model
     * @param finetuned models fine-tuned from it
     * @param density the share of each task vector kept
     * @param lambda the scale of the merged task vector
     * @return a new model with the merged weights
     */
    public static Model ties(Model base, List<Model> finetuned, double density, double lambda) {
        double[] b = ParameterVector.flatten(base.parameters());
        List<double[]> f = finetuned.stream().map(m -> ParameterVector.flatten(m.parameters())).toList();
        Model out = base.copy(base.device());
        ParameterVector.assign(out.parameters(), ties(b, f, density, lambda));
        return out;
    }

    private static void check(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("parameter vectors differ in length: " + a.length + " vs " + b.length);
        }
    }
}
