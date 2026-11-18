package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * Metrics that judge an attribution method rather than a model.
 *
 * <ul>
 *   <li>{@link Metric#ROAR} and {@link Metric#KAR}, remove-and-retrain and keep-and-retrain
 *       (Hooker, Erhan, Kindermans and Kim, NeurIPS 2019, <a href="https://arxiv.org/abs/1806.10758">arXiv:1806.10758</a>):
 *       the top (or all but the top) {@code k%} attributed features of every training and test
 *       row are replaced by an uninformative fill value, the model is retrained, and the test
 *       accuracy is measured; a better attribution makes ROAR accuracy fall faster and KAR
 *       accuracy stay higher. Retraining keeps the masked inputs in distribution, which a deletion
 *       curve without retraining ({@link #deletionAuc}) does not.</li>
 *   <li>{@link Metric#INFIDELITY} (Yeh, Hsieh, Suggala, Inouye and Ravikumar, NeurIPS 2019,
 *       <a href="https://arxiv.org/abs/1901.09392">arXiv:1901.09392</a>):
 *       {@code E_I[(I . phi - (f(x) - f(x - I)))^2]} for Gaussian perturbations {@code I}, with
 *       the attribution rescaled by the least-squares optimal factor as in the authors' code, so
 *       that methods of different scale compare; reported relative to {@code E[(f(x) - f(x - I))^2]}.</li>
 *   <li>{@link Metric#MAX_SENSITIVITY} (same paper): {@code max ||phi(x + d) - phi(x)|| / ||phi(x)||}
 *       over perturbations {@code d} drawn uniformly from an L-infinity ball (normalised by the
 *       attribution's norm so that methods of different scale compare).</li>
 *   <li>{@link Metric#MODEL_RANDOMISATION} and {@link Metric#DATA_RANDOMISATION}, the sanity
 *       checks (Adebayo, Gilmer, Muelly, Goodfellow, Hardt and Kim, NeurIPS 2018, <a href="https://arxiv.org/abs/1810.03292">arXiv:1810.03292</a>):
 *       {@link #cascade} re-draws the top parameter tensors of a trained model (cascading from the
 *       output), {@link #shuffledLabels} permutes the labels for retraining; a method passes when
 *       its maps change, measured by {@link #rankCorrelation} and {@link #ssim} of the absolute
 *       maps before and after.</li>
 * </ul>
 */
public final class AttributionEvaluation {

    /** The evaluation protocols. */
    public enum Metric {
        /** Remove the top-attributed features and retrain. */
        ROAR,
        /** Keep only the top-attributed features and retrain. */
        KAR,
        /** Expected squared error of the attribution as a first-order model of perturbations. */
        INFIDELITY,
        /** Worst relative change of the attribution under small input perturbations. */
        MAX_SENSITIVITY,
        /** Map similarity after cascading re-initialisation of the model's parameters. */
        MODEL_RANDOMISATION,
        /** Map similarity after retraining on permuted labels. */
        DATA_RANDOMISATION
    }

    /** Retrains a classifier on masked data and returns its class-1 probability. */
    @FunctionalInterface
    public interface Retrainer {

        /**
         * @param x the masked training rows
         * @param y the 0/1 labels
         * @return the trained model's class-1 probability as a batch function
         */
        Function<double[][], double[]> fit(double[][] x, double[] y);
    }

    private AttributionEvaluation() {
    }

    // ------------------------------------------------------------------ ROAR / KAR

    /**
     * Replaces the top (or, when {@code removeTop} is false, all but the top) {@code fraction} of
     * every row's features, ranked by absolute attribution, by the fill value.
     *
     * @param x rows
     * @param attributions one attribution per row
     * @param fraction the share of features ranked as top
     * @param removeTop true for ROAR, false for KAR
     * @param fill the per-feature fill value (such as the training mean)
     * @return masked copies
     */
    public static double[][] mask(double[][] x, double[][] attributions, double fraction, boolean removeTop,
            double[] fill) {
        int d = x[0].length;
        int k = (int) Math.round(fraction * d);
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            int[] order = byMagnitude(attributions[i]);
            out[i] = x[i].clone();
            for (int r = 0; r < d; r++) {
                boolean top = r < k;
                if (top == removeTop) {
                    out[i][order[r]] = fill[order[r]];
                }
            }
        }
        return out;
    }

    /**
     * ROAR or KAR: test accuracy after masking and retraining, per fraction.
     *
     * @param metric {@link Metric#ROAR} or {@link Metric#KAR}
     * @param trainX training rows
     * @param trainY training labels (0/1)
     * @param trainAttr attributions of the training rows (from the original model)
     * @param testX test rows
     * @param testY test labels
     * @param testAttr attributions of the test rows
     * @param fill the per-feature fill value
     * @param fractions the shares of features ranked as top
     * @param retrainer fits a fresh model
     * @return the test accuracy per fraction
     */
    public static double[] removeAndRetrain(Metric metric, double[][] trainX, double[] trainY, double[][] trainAttr,
            double[][] testX, double[] testY, double[][] testAttr, double[] fill, double[] fractions,
            Retrainer retrainer) {
        if (metric != Metric.ROAR && metric != Metric.KAR) {
            throw new IllegalArgumentException("ROAR or KAR, not " + metric);
        }
        boolean remove = metric == Metric.ROAR;
        double[] out = new double[fractions.length];
        for (int f = 0; f < fractions.length; f++) {
            Function<double[][], double[]> model = retrainer.fit(mask(trainX, trainAttr, fractions[f], remove, fill),
                    trainY);
            double[] p = model.apply(mask(testX, testAttr, fractions[f], remove, fill));
            int ok = 0;
            for (int i = 0; i < p.length; i++) {
                ok += (p[i] > 0.5 ? 1 : 0) == (int) testY[i] ? 1 : 0;
            }
            out[f] = ok / (double) p.length;
        }
        return out;
    }

    /**
     * The deletion curve without retraining: features are reset to the fill value in order of
     * decreasing absolute attribution, in {@code steps} equal chunks, and the score of the class
     * predicted at {@code x} is averaged along the curve (lower is better).
     *
     * @param score the predicted class's probability as a batch function
     * @param x the row
     * @param attribution its attribution
     * @param fill the fill value per feature
     * @param steps the number of deletion steps
     * @return the area under the deletion curve
     */
    public static double deletionAuc(Function<double[][], double[]> score, double[] x, double[] attribution,
            double[] fill, int steps) {
        int d = x.length;
        int[] order = byMagnitude(attribution);
        double[][] curve = new double[steps + 1][];
        double[] cur = x.clone();
        curve[0] = cur.clone();
        for (int s = 1; s <= steps; s++) {
            int from = (int) ((long) (s - 1) * d / steps);
            int to = (int) ((long) s * d / steps);
            for (int r = from; r < to; r++) {
                cur[order[r]] = fill[order[r]];
            }
            curve[s] = cur.clone();
        }
        return Arrays.stream(score.apply(curve)).average().orElse(0);
    }

    // ------------------------------------------------------------------ infidelity, sensitivity

    /**
     * Infidelity with Gaussian perturbations, the attribution rescaled by its optimal factor,
     * relative to the mean squared output change.
     *
     * @param model the explained output as a batch function
     * @param x the row
     * @param attribution its attribution
     * @param sigma the perturbation standard deviation per feature
     * @param samples the perturbations drawn
     * @param random draws them
     * @return the relative infidelity, 0 for a perfect first-order model, 1 for a useless one
     */
    public static double infidelity(Function<double[][], double[]> model, double[] x, double[] attribution,
            double sigma, int samples, SplittableRandom random) {
        int d = x.length;
        double[][] rows = new double[samples + 1][];
        double[] dot = new double[samples];
        rows[0] = x;
        for (int s = 0; s < samples; s++) {
            double[] row = x.clone();
            for (int j = 0; j < d; j++) {
                double e = sigma * random.nextGaussian();
                row[j] -= e;
                dot[s] += e * attribution[j];
            }
            rows[s + 1] = row;
        }
        double[] f = model.apply(rows);
        double[] delta = new double[samples];
        double sxy = 0;
        double sxx = 0;
        double syy = 0;
        for (int s = 0; s < samples; s++) {
            delta[s] = f[0] - f[s + 1];
            sxy += dot[s] * delta[s];
            sxx += dot[s] * dot[s];
            syy += delta[s] * delta[s];
        }
        if (syy == 0) {
            return 0;
        }
        double beta = sxx > 0 ? sxy / sxx : 0;
        double err = 0;
        for (int s = 0; s < samples; s++) {
            double r = beta * dot[s] - delta[s];
            err += r * r;
        }
        return err / syy;
    }

    /**
     * Max-sensitivity: the largest relative change of the attribution over perturbations drawn
     * uniformly from the L-infinity ball of the radius.
     *
     * @param explainer the attribution method
     * @param x the row
     * @param radius the ball radius
     * @param samples the perturbations drawn
     * @param random draws them
     * @return {@code max ||phi(x + d) - phi(x)|| / ||phi(x)||}
     */
    public static double maxSensitivity(Function<double[], double[]> explainer, double[] x, double radius, int samples,
            SplittableRandom random) {
        double[] base = explainer.apply(x);
        double norm = Math.sqrt(Arrays.stream(base).map(v -> v * v).sum());
        if (norm == 0) {
            return 0;
        }
        double worst = 0;
        for (int s = 0; s < samples; s++) {
            double[] z = x.clone();
            for (int j = 0; j < z.length; j++) {
                z[j] += radius * (2 * random.nextDouble() - 1);
            }
            double[] a = explainer.apply(z);
            double diff = 0;
            for (int j = 0; j < a.length; j++) {
                diff += (a[j] - base[j]) * (a[j] - base[j]);
            }
            worst = Math.max(worst, Math.sqrt(diff) / norm);
        }
        return worst;
    }

    // ------------------------------------------------------------------ sanity checks

    /**
     * Cascading model randomisation: a fresh model of the same architecture carrying the trained
     * parameters, except the top {@code top} parameter tensors (counted from the output), which
     * are re-drawn from a Gaussian with the trained tensor's standard deviation (0.1 when it is
     * constant, as zero-initialised biases are).
     *
     * @param trained the trained model
     * @param architecture builds an untrained model of the same architecture
     * @param top how many parameter tensors to re-draw, from the output side
     * @param seed the seed
     * @return the partly randomised model
     */
    public static Model cascade(Model trained, Supplier<Model> architecture, int top, long seed) {
        Model fresh = architecture.get();
        List<Var> from = trained.parameters();
        List<Var> to = fresh.parameters();
        if (from.size() != to.size()) {
            throw new IllegalArgumentException("architectures differ");
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < from.size(); i++) {
            Tensor v = from.get(i).value();
            if (i < from.size() - top) {
                to.get(i).assign(v);
            } else {
                float[] a = v.toArray();
                double mean = 0;
                for (float f : a) {
                    mean += f / (double) a.length;
                }
                double var = 0;
                for (float f : a) {
                    var += (f - mean) * (f - mean) / a.length;
                }
                float std = var > 0 ? (float) Math.sqrt(var) : 0.1f;
                to.get(i).assign(Tensor.normal(v.shape(), 0f, std, r));
            }
        }
        return fresh;
    }

    /**
     * Labels permuted at random, for the data-randomisation test.
     *
     * @param y labels
     * @param seed the seed
     * @return a permuted copy
     */
    public static int[] shuffledLabels(int[] y, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] out = y.clone();
        for (int i = out.length - 1; i > 0; i--) {
            int k = r.nextInt(i + 1);
            int t = out[i];
            out[i] = out[k];
            out[k] = t;
        }
        return out;
    }

    /**
     * Spearman rank correlation of the absolute values of two maps.
     *
     * @param a a map
     * @param b another map
     * @return the correlation
     */
    public static double rankCorrelation(double[] a, double[] b) {
        double[] ra = ranks(Arrays.stream(a).map(Math::abs).toArray());
        double[] rb = ranks(Arrays.stream(b).map(Math::abs).toArray());
        double ma = Arrays.stream(ra).average().orElse(0);
        double mb = Arrays.stream(rb).average().orElse(0);
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < ra.length; i++) {
            sab += (ra[i] - ma) * (rb[i] - mb);
            saa += (ra[i] - ma) * (ra[i] - ma);
            sbb += (rb[i] - mb) * (rb[i] - mb);
        }
        return saa == 0 || sbb == 0 ? 0 : sab / Math.sqrt(saa * sbb);
    }

    /**
     * The structural similarity of the absolute maps, each scaled to {@code [0, 1]}, in one
     * window over the whole map (the constants of Wang et al. 2004 with dynamic range 1).
     *
     * @param a a map
     * @param b another map
     * @return SSIM in {@code [-1, 1]}
     */
    public static double ssim(double[] a, double[] b) {
        double[] x = unit(a);
        double[] y = unit(b);
        int n = x.length;
        double mx = Arrays.stream(x).average().orElse(0);
        double my = Arrays.stream(y).average().orElse(0);
        double vx = 0;
        double vy = 0;
        double cxy = 0;
        for (int i = 0; i < n; i++) {
            vx += (x[i] - mx) * (x[i] - mx) / n;
            vy += (y[i] - my) * (y[i] - my) / n;
            cxy += (x[i] - mx) * (y[i] - my) / n;
        }
        double c1 = 1e-4;
        double c2 = 9e-4;
        return (2 * mx * my + c1) * (2 * cxy + c2) / ((mx * mx + my * my + c1) * (vx + vy + c2));
    }

    private static double[] unit(double[] a) {
        double max = Arrays.stream(a).map(Math::abs).max().orElse(0);
        return Arrays.stream(a).map(v -> max == 0 ? 0 : Math.abs(v) / max).toArray();
    }

    private static double[] ranks(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (i, j) -> Double.compare(v[i], v[j]));
        double[] r = new double[v.length];
        int i = 0;
        while (i < v.length) {
            int j = i;
            while (j + 1 < v.length && v[idx[j + 1]] == v[idx[i]]) {
                j++;
            }
            double avg = (i + j) / 2.0;
            for (int k = i; k <= j; k++) {
                r[idx[k]] = avg;
            }
            i = j + 1;
        }
        return r;
    }

    private static int[] byMagnitude(double[] a) {
        return IntStream.range(0, a.length).boxed().sorted((i, j) -> Double.compare(Math.abs(a[j]), Math.abs(a[i])))
                .mapToInt(Integer::intValue).toArray();
    }
}
