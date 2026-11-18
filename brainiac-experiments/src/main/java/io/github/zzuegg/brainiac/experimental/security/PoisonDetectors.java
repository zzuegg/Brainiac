package io.github.zzuegg.brainiac.experimental.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.ComponentAnalysis;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.nn.LayerHooks;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Detectors of poisoned training samples from a trained (backdoored) model's
 * representations. Every detector returns a suspicion score per sample, higher meaning more
 * likely poisoned, so detection AUROC and recall at a removal budget can be compared:
 *
 * <ul>
 *   <li>{@link #spectral}: spectral signatures score each sample of a class by its squared
 *       projection on the top singular vector of the class's centred penultimate features
 *       divided by the class's median score so that classes compare
 *       (Tran, Li and Madry, NeurIPS 2018, <a href="https://arxiv.org/abs/1811.00636">arXiv:1811.00636</a>).</li>
 *   <li>{@link #activationClustering}: per class, FastICA to a few components and 2-means;
 *       a small cluster (under 35% of the class) is suspicious, scored by how much closer a
 *       sample is to it than to the large one (Chen et al. 2018,
 *       <a href="https://arxiv.org/abs/1811.03728">arXiv:1811.03728</a>).</li>
 *   <li>{@link #agpd}: activation-gradient poisoned-sample detection (Yuan et al., ICLR
 *       2025): the gradient of the sample's own logit with respect to a hidden activation,
 *       its circular (angle) distribution against a clean reference, the CVBT dispersion
 *       per class with a robust z-score ({@code z >= e²}) to find the target class, and the
 *       closeness score {@code s = (1 - cos θ_far) / ((1 - cos θ_far) + (1 - cos θ_ref))}
 *       with iterative removal at {@code s < τ_s} inside it. The clean reference of each
 *       class is the mean gradient of a few clean samples; the stopping rule is the paper's
 *       sliding-window minimum of the Jensen-Shannon change between rounds.</li>
 * </ul>
 */
public final class PoisonDetectors {

    private PoisonDetectors() {
    }

    /**
     * AGPD's output.
     *
     * @param dispersion the CVBT dispersion of each class
     * @param zScores the robust z-score of each class
     * @param targets the classes flagged as backdoor targets
     * @param suspicion one score per sample ({@code 1 - s} inside flagged classes, below 0
     *     elsewhere)
     * @param flagged the samples reported as poisoned
     */
    public record Agpd(double[] dispersion, double[] zScores, int[] targets, double[] suspicion, boolean[] flagged) {
    }

    /**
     * A hidden layer's activations.
     *
     * @param model the model
     * @param x the inputs {@code [n, features]}
     * @param layer the step index
     * @return one row per sample
     */
    public static double[][] features(Model model, float[][] x, int layer) {
        Tensor h = LayerHooks.activations(model, Tensor.of(x)).get(layer);
        return ThinSvd.toMatrix(h);
    }

    /**
     * Activation gradients: {@code ∂ z_y / ∂ h} for each sample's own label {@code y}, at the
     * output of a hidden step.
     *
     * @param model the model
     * @param x the inputs
     * @param y the (training) labels
     * @param layer the step index
     * @return one gradient row per sample
     */
    public static double[][] activationGradients(Model model, float[][] x, int[] y, int layer) {
        int n = x.length;
        int width = LayerHooks.activations(model, Tensor.of(new float[][] {x[0]})).get(layer).shape().dim(1);
        Var delta = Var.param(Tensor.zeros(Shape.of(n, width)));
        Var logits = model.forward(Var.of(Tensor.of(x)), false,
                (step, out) -> step.index() == layer ? out.add(delta) : out);
        float[] ids = new float[n];
        for (int i = 0; i < n; i++) {
            ids[i] = y[i];
        }
        Var own = logits.takeAlong(Tensor.of(Shape.of(n, 1), ids)).sum();
        Tensor g = Var.gradients(own, List.of(delta), false).get(0).value();
        return ThinSvd.toMatrix(g);
    }

    /**
     * Spectral-signature scores.
     *
     * @param features penultimate features
     * @param y the training labels
     * @param classes the number of classes
     * @return one score per sample
     */
    public static double[] spectral(double[][] features, int[] y, int classes) {
        double[] scores = new double[y.length];
        for (int c = 0; c < classes; c++) {
            int[] rows = rowsOf(y, c);
            if (rows.length < 3) {
                continue;
            }
            double[][] centred = centre(features, rows);
            double[][] v = ThinSvd.of(centred).v();
            double[] proj = new double[rows.length];
            for (int i = 0; i < rows.length; i++) {
                double p = 0;
                for (int j = 0; j < v.length; j++) {
                    p += centred[i][j] * v[j][0];
                }
                proj[i] = p * p;
            }
            double scale = Math.max(median(proj), 1e-12);
            for (int i = 0; i < rows.length; i++) {
                scores[rows[i]] = proj[i] / scale;
            }
        }
        return scores;
    }

    /**
     * Activation-clustering scores.
     *
     * @param features penultimate features
     * @param y the training labels
     * @param classes the number of classes
     * @param components ICA components, such as 3
     * @param seed the seed
     * @return one score per sample
     */
    public static double[] activationClustering(double[][] features, int[] y, int classes, int components,
            long seed) {
        double[] scores = new double[y.length];
        Arrays.fill(scores, -10);
        for (int c = 0; c < classes; c++) {
            int[] rows = rowsOf(y, c);
            if (rows.length < 6) {
                continue;
            }
            double[][] sub = new double[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                sub[i] = features[rows[i]];
            }
            double[][] z = ComponentAnalysis.fastIca(sub, components, 100, seed + c).transform(sub);
            KMeans km = KMeans.fit(z, 2, KMeans.Init.PLUS_PLUS, 3, 50, seed + c);
            int[] count = new int[2];
            for (int l : km.labels()) {
                count[l]++;
            }
            int small = count[0] <= count[1] ? 0 : 1;
            double share = count[small] / (double) rows.length;
            double offset = share < 0.35 ? 0 : -5;
            for (int i = 0; i < rows.length; i++) {
                double ds = Math.sqrt(dist2(z[i], km.centres()[small]));
                double dl = Math.sqrt(dist2(z[i], km.centres()[1 - small]));
                scores[rows[i]] = offset + (dl - ds) / (dl + ds + 1e-12);
            }
        }
        return scores;
    }

    /**
     * AGPD.
     *
     * @param gradients activation gradients of the training samples
     * @param y the training labels
     * @param classes the number of classes
     * @param references a clean reference gradient per class
     * @return the detection
     */
    public static Agpd agpd(double[][] gradients, int[] y, int classes, double[][] references) {
        double[] rho = new double[classes];
        for (int c = 0; c < classes; c++) {
            int[] rows = rowsOf(y, c);
            if (rows.length < 2) {
                continue;
            }
            double[] cos0 = new double[rows.length];
            int far = 0;
            for (int i = 0; i < rows.length; i++) {
                cos0[i] = cosine(gradients[rows[i]], references[c]);
                far = cos0[i] < cos0[far] ? i : far;
            }
            double s = 0;
            for (int i = 0; i < rows.length; i++) {
                double d = cos0[i] - cosine(gradients[rows[i]], gradients[rows[far]]);
                s += d * d / rows.length;
            }
            rho[c] = Math.sqrt(s);
        }
        double med = median(rho);
        double[] dev = new double[classes];
        for (int c = 0; c < classes; c++) {
            dev[c] = Math.abs(rho[c] - med);
        }
        double mad = Math.max(1.4826 * median(dev), 1e-9);
        double[] z = new double[classes];
        List<Integer> targets = new ArrayList<>();
        for (int c = 0; c < classes; c++) {
            z[c] = (rho[c] - med) / mad;
            if (z[c] >= Math.exp(2)) {
                targets.add(c);
            }
        }
        double[] suspicion = new double[y.length];
        Arrays.fill(suspicion, -1);
        boolean[] flagged = new boolean[y.length];
        for (int c : targets) {
            filter(gradients, rowsOf(y, c), references[c], suspicion, flagged);
        }
        return new Agpd(rho, z, targets.stream().mapToInt(Integer::intValue).toArray(), suspicion, flagged);
    }

    private static void filter(double[][] g, int[] rows, double[] ref, double[] suspicion, boolean[] flagged) {
        List<Integer> remaining = new ArrayList<>();
        for (int r : rows) {
            remaining.add(r);
        }
        List<List<Integer>> removedPerRound = new ArrayList<>();
        List<Double> js = new ArrayList<>();
        double[] previous = null;
        boolean first = true;
        while (!remaining.isEmpty()) {
            int far = remaining.get(0);
            for (int r : remaining) {
                far = cosine(g[r], ref) < cosine(g[far], ref) ? r : far;
            }
            double[] hist = new double[20];
            List<Integer> removed = new ArrayList<>();
            List<Integer> kept = new ArrayList<>();
            for (int r : remaining) {
                double a = 1 - cosine(g[r], g[far]);
                double b = 1 - cosine(g[r], ref);
                double s = a / (a + b + 1e-12);
                if (first) {
                    suspicion[r] = 1 - s;
                }
                hist[Math.min(19, (int) (s * 20))] += 1.0 / remaining.size();
                if (s < 0.05 || r == far) {
                    removed.add(r);
                } else {
                    kept.add(r);
                }
            }
            first = false;
            js.add(previous == null ? 0 : jensenShannon(previous, hist));
            previous = hist;
            removedPerRound.add(removed);
            remaining = kept;
        }
        int w = 10;
        int stop = removedPerRound.size();
        if (js.size() > w) {
            double best = Double.POSITIVE_INFINITY;
            int bestM = 0;
            for (int m = 0; m + w <= js.size(); m++) {
                double mu = 0;
                for (int j = 0; j < w; j++) {
                    mu += js.get(m + j) / w;
                }
                double var = 0;
                for (int j = 0; j < w; j++) {
                    var += (js.get(m + j) - mu) * (js.get(m + j) - mu) / w;
                }
                double score = mu + 5 * Math.sqrt(var);
                if (score < best) {
                    best = score;
                    bestM = m;
                }
            }
            stop = bestM + w / 2;
        } else {
            stop = Math.max(1, stop / 2);
        }
        for (int t = 0; t < Math.min(stop, removedPerRound.size()); t++) {
            for (int r : removedPerRound.get(t)) {
                flagged[r] = true;
            }
        }
    }

    /**
     * Loss-based filtering scores: the per-sample training loss (the baseline that drops the
     * highest-loss samples).
     *
     * @param model the model
     * @param x the inputs
     * @param y the training labels
     * @return the cross-entropy of each sample's label
     */
    public static double[] lossScores(Model model, float[][] x, int[] y) {
        float[] z = model.predict(Tensor.of(x)).toArray();
        int k = z.length / x.length;
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < k; c++) {
                max = Math.max(max, z[i * k + c]);
            }
            double s = 0;
            for (int c = 0; c < k; c++) {
                s += Math.exp(z[i * k + c] - max);
            }
            out[i] = Math.log(s) + max - z[i * k + y[i]];
        }
        return out;
    }

    /**
     * The rows with the highest scores.
     *
     * @param scores the suspicion scores
     * @param budget how many to remove
     * @return a flag per row
     */
    public static boolean[] top(double[] scores, int budget) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[b], scores[a]));
        boolean[] out = new boolean[scores.length];
        for (int i = 0; i < Math.min(budget, order.length); i++) {
            out[order[i]] = true;
        }
        return out;
    }

    /**
     * Cosine similarity.
     *
     * @param a one vector
     * @param b another
     * @return their cosine
     */
    public static double cosine(double[] a, double[] b) {
        double ab = 0;
        double aa = 0;
        double bb = 0;
        for (int j = 0; j < a.length; j++) {
            ab += a[j] * b[j];
            aa += a[j] * a[j];
            bb += b[j] * b[j];
        }
        return ab / Math.sqrt(Math.max(aa * bb, 1e-24));
    }

    static int[] rowsOf(int[] y, int c) {
        int count = 0;
        for (int v : y) {
            count += v == c ? 1 : 0;
        }
        int[] rows = new int[count];
        int k = 0;
        for (int i = 0; i < y.length; i++) {
            if (y[i] == c) {
                rows[k++] = i;
            }
        }
        return rows;
    }

    private static double[][] centre(double[][] f, int[] rows) {
        int d = f[0].length;
        double[] mean = new double[d];
        for (int r : rows) {
            for (int j = 0; j < d; j++) {
                mean[j] += f[r][j] / rows.length;
            }
        }
        double[][] out = new double[rows.length][d];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = f[rows[i]][j] - mean[j];
            }
        }
        return out;
    }

    private static double dist2(double[] a, double[] b) {
        double s = 0;
        for (int j = 0; j < a.length; j++) {
            s += (a[j] - b[j]) * (a[j] - b[j]);
        }
        return s;
    }

    private static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        int n = s.length;
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }

    private static double jensenShannon(double[] p, double[] q) {
        double s = 0;
        for (int i = 0; i < p.length; i++) {
            double m = 0.5 * (p[i] + q[i]);
            if (p[i] > 0) {
                s += 0.5 * p[i] * Math.log(p[i] / m);
            }
            if (q[i] > 0) {
                s += 0.5 * q[i] * Math.log(q[i] / m);
            }
        }
        return s;
    }
}
