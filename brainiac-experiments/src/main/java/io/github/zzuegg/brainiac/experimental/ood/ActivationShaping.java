package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Activation shaping for OOD detection: the penultimate activations of a
 * {@link FeatureClassifier} are reshaped before its head, and the energy of the shaped logits
 * is the score (higher means "more in-distribution").
 *
 * <ul>
 *   <li>{@link Method#REACT}: clip every activation at a percentile of the training
 *       activations (Sun et al., NeurIPS 2021, <a href="https://arxiv.org/abs/2111.12797">arXiv:2111.12797</a>).</li>
 *   <li>{@link Method#ASH_P}, {@link Method#ASH_B}, {@link Method#ASH_S}: keep a sample's top
 *       activations above its own {@code p}-th percentile and zero the rest; P keeps them as
 *       they are, B replaces them by the total sum spread evenly, S scales them by
 *       {@code exp(s1 / s2)} (total over kept sum) (Djurisic et al., ICLR 2023,
 *       <a href="https://arxiv.org/abs/2209.09858">arXiv:2209.09858</a>).</li>
 *   <li>{@link Method#SCALE}: ASH-S's scaling without the pruning, which leaves the predicted
 *       class almost unchanged (Xu et al., ICLR 2024, <a href="https://arxiv.org/abs/2310.00227">arXiv:2310.00227</a>).</li>
 *   <li>{@link Method#ADASCALE_A}, {@link Method#ADASCALE_L}: a per-sample percentile (Regmi,
 *       2025, <a href="https://arxiv.org/abs/2503.08023">arXiv:2503.08023</a>). The 5% of input
 *       values with the smallest attribution (gradient of the predicted logit) are moved by
 *       {@code epsilon sign(attribution)}; the shift of the top 1% activations
 *       {@code Q = sum |a' - a|} plus the mass of the perturbed top 5%
 *       {@code C = sum relu(a')} gives {@code Q' = lambda Q + C}, larger for OOD inputs. Its
 *       empirical CDF on a few in-distribution validation inputs sets
 *       {@code p = pMin + (1 - F(Q')) (pMax - pMin)}, so likely-ID inputs are scaled more. A
 *       scales the activations by {@code exp(r)} with {@code r = s1 / s_p}; L multiplies the
 *       logits by {@code r^2}. It costs one backward and one extra forward pass.</li>
 * </ul>
 */
public final class ActivationShaping {

    /** The shaping. */
    public enum Method {
        /** the plain energy score */
        NONE,
        /** clipping at a training percentile */
        REACT,
        /** prune, keep values */
        ASH_P,
        /** prune, binarise */
        ASH_B,
        /** prune, scale */
        ASH_S,
        /** scale without pruning */
        SCALE,
        /** adaptive-percentile scaling of the activations */
        ADASCALE_A,
        /** adaptive-percentile scaling of the logits */
        ADASCALE_L
    }

    private final FeatureClassifier classifier;
    private final float reactClip;
    private final Tensor validation;
    private double[] validationQ;
    private double ashPercentile = 0.85;
    private double scalePercentile = 0.85;
    private double pMin = 0.6;
    private double pMax = 0.9;
    private float epsilon = 0.5f;
    private double lambda = 10;

    private ActivationShaping(FeatureClassifier classifier, float reactClip, Tensor validation) {
        this.classifier = classifier;
        this.reactClip = reactClip;
        this.validation = validation;
    }

    /**
     * Fits the shaping statistics.
     *
     * @param classifier the trained classifier
     * @param train in-distribution training inputs (ReAct's percentile)
     * @param validation a few in-distribution inputs (AdaSCALE's CDF; about 10 suffice)
     * @param reactPercentile ReAct's clipping percentile, such as 0.9
     * @return the shaper
     */
    public static ActivationShaping fit(FeatureClassifier classifier, Tensor train, Tensor validation,
            double reactPercentile) {
        Objects.requireNonNull(classifier);
        float[] a = classifier.features(train).toArray();
        Arrays.sort(a);
        float clip = a[(int) Math.min(a.length - 1, Math.floor(reactPercentile * a.length))];
        return new ActivationShaping(classifier, clip, Objects.requireNonNull(validation));
    }

    /**
     * @param ash ASH's percentile (share pruned), such as 0.85
     * @param scale SCALE's percentile, such as 0.85
     * @return this
     */
    public ActivationShaping percentiles(double ash, double scale) {
        this.ashPercentile = ash;
        this.scalePercentile = scale;
        return this;
    }

    /**
     * @param min AdaSCALE's lowest percentile, such as 0.6
     * @param max its highest percentile, such as 0.9
     * @param eps the input perturbation, such as 0.5
     * @param shiftWeight {@code lambda}, such as 10
     * @return this
     */
    public ActivationShaping adaptive(double min, double max, float eps, double shiftWeight) {
        this.pMin = min;
        this.pMax = max;
        this.epsilon = eps;
        this.lambda = shiftWeight;
        this.validationQ = null;
        return this;
    }

    /** @return ReAct's clipping threshold */
    public float reactClip() {
        return reactClip;
    }

    /**
     * The shaped logits.
     *
     * @param method the shaping
     * @param x the inputs
     * @return the logits after shaping
     */
    public Tensor logits(Method method, Tensor x) {
        Tensor h = classifier.features(x);
        int n = h.shape().dim(0);
        int d = h.shape().dim(1);
        float[] v = h.toArray();
        switch (method) {
            case NONE -> {
                return head(h);
            }
            case REACT -> {
                for (int i = 0; i < v.length; i++) {
                    v[i] = Math.min(v[i], reactClip);
                }
            }
            case ASH_P, ASH_B, ASH_S -> {
                for (int i = 0; i < n; i++) {
                    ash(v, i * d, d, ashPercentile, method);
                }
            }
            case SCALE -> {
                for (int i = 0; i < n; i++) {
                    double r = ratio(v, i * d, d, scalePercentile);
                    scaleRow(v, i * d, d, Math.exp(r));
                }
            }
            case ADASCALE_A, ADASCALE_L -> {
                double[] q = oodness(x);
                double[] r = new double[n];
                for (int i = 0; i < n; i++) {
                    double cdf = cdf(q[i]);
                    double p = pMin + (1 - cdf) * (pMax - pMin);
                    r[i] = ratio(v, i * d, d, p);
                    if (method == Method.ADASCALE_A) {
                        scaleRow(v, i * d, d, Math.exp(r[i]));
                    }
                }
                if (method == Method.ADASCALE_L) {
                    float[] z = head(h).toArray();
                    int c = z.length / n;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < c; j++) {
                            z[i * c + j] *= (float) (r[i] * r[i]);
                        }
                    }
                    return Tensor.of(Shape.of(n, c), z);
                }
            }
        }
        return head(Tensor.of(h.shape(), v));
    }

    /**
     * The OOD score: the energy of the shaped logits.
     *
     * @param method the shaping
     * @param x the inputs
     * @return {@code logsumexp} of the shaped logits, higher meaning in-distribution
     */
    public double[] score(Method method, Tensor x) {
        return LogitScores.energy(logits(method, x), 1f);
    }

    /**
     * AdaSCALE's OOD-ness statistic {@code Q' = lambda Q + C}.
     *
     * @param x the inputs
     * @return per input, larger for likely-OOD inputs
     */
    public double[] oodness(Tensor x) {
        int n = x.shape().dim(0);
        int in = Math.toIntExact(x.size() / n);
        Var xv = Var.param(x);
        Var feats = classifier.features(xv, false);
        Var logits = classifier.head().forward(feats);
        Tensor oneHot = oneHotArgmax(logits.value());
        Tensor attr = Var.gradients(logits.mul(oneHot).sum(), List.of(xv), false).getFirst().value();
        float[] g = attr.toArray();
        float[] xs = x.toArray();
        int pick = Math.max(1, (int) Math.round(0.05 * in));
        for (int i = 0; i < n; i++) {
            Integer[] order = new Integer[in];
            for (int j = 0; j < in; j++) {
                order[j] = j;
            }
            int row = i * in;
            Arrays.sort(order, (p, q) -> Float.compare(Math.abs(g[row + p]), Math.abs(g[row + q])));
            for (int t = 0; t < pick; t++) {
                int j = order[t];
                xs[row + j] += epsilon * Math.signum(g[row + j]);
            }
        }
        float[] a = feats.value().toArray();
        float[] ap = classifier.features(Tensor.of(x.shape(), xs)).toArray();
        int d = a.length / n;
        int k1 = Math.max(1, (int) Math.round(0.01 * d));
        int k2 = Math.max(1, (int) Math.round(0.05 * d));
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            int row = i * d;
            Integer[] order = new Integer[d];
            for (int j = 0; j < d; j++) {
                order[j] = j;
            }
            Arrays.sort(order, (p, q) -> Float.compare(Math.abs(a[row + q]), Math.abs(a[row + p])));
            double shift = 0;
            for (int t = 0; t < k1; t++) {
                shift += Math.abs(ap[row + order[t]] - a[row + order[t]]);
            }
            float[] pert = Arrays.copyOfRange(ap, row, row + d);
            Arrays.sort(pert);
            double mass = 0;
            for (int t = 0; t < k2; t++) {
                mass += Math.max(0, pert[d - 1 - t]);
            }
            out[i] = lambda * shift + mass;
        }
        return out;
    }

    private double cdf(double q) {
        if (validationQ == null) {
            validationQ = oodness(validation);
            Arrays.sort(validationQ);
        }
        int idx = Arrays.binarySearch(validationQ, q);
        int below = idx >= 0 ? idx + 1 : -idx - 1;
        return below / (double) validationQ.length;
    }

    private Tensor head(Tensor h) {
        return Var.noGrad(() -> classifier.head().forward(Var.of(h)).value());
    }

    /** {@code r = s1 / (sum of activations above the p-th percentile of the row)}. */
    private static double ratio(float[] v, int off, int d, double p) {
        float[] row = Arrays.copyOfRange(v, off, off + d);
        double s1 = 0;
        for (float f : row) {
            s1 += f;
        }
        Arrays.sort(row);
        int cut = (int) Math.min(d - 1, Math.floor(p * d));
        double s2 = 0;
        for (int j = cut; j < d; j++) {
            s2 += row[j];
        }
        return s2 > 1e-12 ? s1 / s2 : 1;
    }

    private static void scaleRow(float[] v, int off, int d, double factor) {
        for (int j = 0; j < d; j++) {
            v[off + j] *= (float) factor;
        }
    }

    private static void ash(float[] v, int off, int d, double p, Method m) {
        float[] row = Arrays.copyOfRange(v, off, off + d);
        double s1 = 0;
        for (float f : row) {
            s1 += f;
        }
        float[] sorted = row.clone();
        Arrays.sort(sorted);
        int cut = (int) Math.min(d - 1, Math.floor(p * d));
        float threshold = sorted[cut];
        int kept = d - cut;
        double s2 = 0;
        for (int j = cut; j < d; j++) {
            s2 += sorted[j];
        }
        for (int j = 0; j < d; j++) {
            boolean keep = row[j] >= threshold;
            float out;
            if (!keep) {
                out = 0;
            } else {
                out = switch (m) {
                    case ASH_B -> (float) (s1 / kept);
                    case ASH_S -> (float) (row[j] * Math.exp(s2 > 1e-12 ? s1 / s2 : 1));
                    default -> row[j];
                };
            }
            v[off + j] = out;
        }
    }

    private static Tensor oneHotArgmax(Tensor logits) {
        int n = logits.shape().dim(0);
        int c = logits.shape().dim(1);
        int[] arg = logits.argmax(1).toIntArray();
        float[] v = new float[n * c];
        for (int i = 0; i < n; i++) {
            v[i * c + arg[i]] = 1;
        }
        return Tensor.of(Shape.of(n, c), v);
    }
}
