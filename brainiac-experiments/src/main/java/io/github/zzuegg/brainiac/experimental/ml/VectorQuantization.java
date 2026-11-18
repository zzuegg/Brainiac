package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * Learning vector quantisation: classification by the nearest labelled prototype.
 * <ul>
 *   <li>{@link Kind#LVQ1} (Kohonen 1990): the winner moves towards a sample of its class and away from
 *       one of another class.</li>
 *   <li>{@link Kind#LVQ2_1}: the nearest correct and nearest wrong prototypes are both updated when the
 *       sample falls in a window around the midplane between them
 *       ({@code min(d+/d-, d-/d+) > (1 - w) / (1 + w)}, distances not squared).</li>
 *   <li>{@link Kind#GLVQ} (Sato and Yamada, NeurIPS 1995): stochastic gradient descent of
 *       {@code sum sigmoid(mu)}, {@code mu = (d+ - d-) / (d+ + d-)}, with {@code d+} ({@code d-}) the
 *       squared distance to the nearest prototype of the right (a wrong) class.</li>
 *   <li>{@link Kind#GMLVQ} (Schneider, Biehl and Hammer, Neural Comput. 2009): GLVQ with the adaptive
 *       distance {@code (x - w)^T Omega^T Omega (x - w)}; {@code Lambda = Omega^T Omega} is kept at unit
 *       trace and its diagonal is a feature-relevance profile.</li>
 *   <li>{@link Kind#LOCALISED_GMLVQ}: one {@code Omega} per prototype.</li>
 * </ul>
 * Prototypes start at class-conditional means plus a small jitter; the metric learning rate is a
 * tenth of the prototype rate, and both decay as {@code 1 / (1 + epoch / 10)}. Inputs are expected
 * standardised.
 */
public final class VectorQuantization {

    /** The learning rule. */
    public enum Kind {
        /** Kohonen's LVQ1. */
        LVQ1,
        /** LVQ2.1 with a window. */
        LVQ2_1,
        /** Generalised LVQ (Euclidean). */
        GLVQ,
        /** Generalised matrix LVQ: one global relevance matrix. */
        GMLVQ,
        /** Localised GMLVQ: one relevance matrix per prototype. */
        LOCALISED_GMLVQ
    }

    private final Kind kind;
    private final double[][] prototypes;
    private final int[] labels;
    private final double[][][] omegas;

    private VectorQuantization(Kind kind, double[][] prototypes, int[] labels, double[][][] omegas) {
        this.kind = kind;
        this.prototypes = prototypes;
        this.labels = labels;
        this.omegas = omegas;
    }

    /**
     * Trains.
     *
     * @param kind the rule
     * @param x standardised training points {@code [n][d]}
     * @param y their classes in {@code [0, classes)}
     * @param classes the number of classes
     * @param perClass prototypes per class
     * @param epochs passes over the data
     * @param learningRate the prototype learning rate, such as 0.05
     * @param seed the seed
     * @return the model
     */
    public static VectorQuantization fit(Kind kind, double[][] x, int[] y, int classes, int perClass, int epochs,
            double learningRate, long seed) {
        int n = x.length;
        int d = x[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        int p = classes * perClass;
        double[][] w = new double[p][d];
        int[] lab = new int[p];
        double[][] mean = new double[classes][d];
        double[][] sd = new double[classes][d];
        int[] count = new int[classes];
        for (int i = 0; i < n; i++) {
            count[y[i]]++;
            for (int j = 0; j < d; j++) {
                mean[y[i]][j] += x[i][j];
            }
        }
        for (int c = 0; c < classes; c++) {
            for (int j = 0; j < d; j++) {
                mean[c][j] /= Math.max(1, count[c]);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double e = x[i][j] - mean[y[i]][j];
                sd[y[i]][j] += e * e / Math.max(1, count[y[i]]);
            }
        }
        for (int c = 0; c < classes; c++) {
            for (int k = 0; k < perClass; k++) {
                int idx = c * perClass + k;
                lab[idx] = c;
                for (int j = 0; j < d; j++) {
                    double jitter = perClass > 1 ? 0.5 * Math.sqrt(sd[c][j]) * r.nextGaussian() : 0;
                    w[idx][j] = mean[c][j] + jitter;
                }
            }
        }
        double[][][] omegas = null;
        if (kind == Kind.GMLVQ || kind == Kind.LOCALISED_GMLVQ) {
            int m = kind == Kind.GMLVQ ? 1 : p;
            omegas = new double[m][d][d];
            for (double[][] o : omegas) {
                for (int j = 0; j < d; j++) {
                    o[j][j] = 1 / Math.sqrt(d);
                }
            }
        }
        VectorQuantization model = new VectorQuantization(kind, w, lab, omegas);
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        for (int e = 0; e < epochs; e++) {
            double lr = learningRate / (1 + e / 10.0);
            for (int i = n - 1; i > 0; i--) {
                int s = r.nextInt(i + 1);
                int t = order[i];
                order[i] = order[s];
                order[s] = t;
            }
            for (int i : order) {
                model.step(x[i], y[i], lr);
            }
        }
        return model;
    }

    private double[][] omegaOf(int prototype) {
        if (omegas == null) {
            return null;
        }
        return omegas.length == 1 ? omegas[0] : omegas[prototype];
    }

    /** {@code Omega (x - w)}, or {@code x - w} without a metric. */
    private double[] projected(double[] x, int prototype) {
        double[] diff = new double[x.length];
        double[] w = prototypes[prototype];
        for (int j = 0; j < x.length; j++) {
            diff[j] = x[j] - w[j];
        }
        double[][] o = omegaOf(prototype);
        if (o == null) {
            return diff;
        }
        double[] out = new double[o.length];
        for (int a = 0; a < o.length; a++) {
            double s = 0;
            for (int j = 0; j < diff.length; j++) {
                s += o[a][j] * diff[j];
            }
            out[a] = s;
        }
        return out;
    }

    private double distance(double[] x, int prototype) {
        double[] v = projected(x, prototype);
        double s = 0;
        for (double e : v) {
            s += e * e;
        }
        return s;
    }

    private void step(double[] x, int y, double lr) {
        int plus = -1;
        int minus = -1;
        double dPlus = Double.POSITIVE_INFINITY;
        double dMinus = Double.POSITIVE_INFINITY;
        int winner = -1;
        double dWin = Double.POSITIVE_INFINITY;
        for (int k = 0; k < prototypes.length; k++) {
            double dk = distance(x, k);
            if (labels[k] == y && dk < dPlus) {
                dPlus = dk;
                plus = k;
            }
            if (labels[k] != y && dk < dMinus) {
                dMinus = dk;
                minus = k;
            }
            if (dk < dWin) {
                dWin = dk;
                winner = k;
            }
        }
        int d = x.length;
        switch (kind) {
            case LVQ1 -> {
                double sign = labels[winner] == y ? 1 : -1;
                for (int j = 0; j < d; j++) {
                    prototypes[winner][j] += sign * lr * (x[j] - prototypes[winner][j]);
                }
            }
            case LVQ2_1 -> {
                if (plus < 0 || minus < 0) {
                    return;
                }
                double a = Math.sqrt(dPlus);
                double b = Math.sqrt(dMinus);
                double window = 0.3;
                if (Math.min(a / Math.max(1e-12, b), b / Math.max(1e-12, a)) > (1 - window) / (1 + window)) {
                    for (int j = 0; j < d; j++) {
                        prototypes[plus][j] += lr * (x[j] - prototypes[plus][j]);
                        prototypes[minus][j] -= lr * (x[j] - prototypes[minus][j]);
                    }
                }
            }
            default -> glvqStep(x, plus, minus, dPlus, dMinus, lr);
        }
    }

    /** One stochastic GLVQ/GMLVQ step on {@code sigmoid(mu)}. */
    private void glvqStep(double[] x, int plus, int minus, double dPlus, double dMinus, double lr) {
        if (plus < 0 || minus < 0) {
            return;
        }
        double sum = dPlus + dMinus + 1e-12;
        double mu = (dPlus - dMinus) / sum;
        double sig = 1 / (1 + Math.exp(-mu));
        double phi = sig * (1 - sig);
        double gPlus = phi * 2 * dMinus / (sum * sum);     // d cost / d dPlus
        double gMinus = -phi * 2 * dPlus / (sum * sum);    // d cost / d dMinus
        int d = x.length;
        double[] vPlus = projected(x, plus);
        double[] vMinus = projected(x, minus);
        double[][] oPlus = omegaOf(plus);
        double[][] oMinus = omegaOf(minus);
        // d dist / d w = -2 Omega^T Omega (x - w)
        double[] dwPlus = lambdaTimes(oPlus, vPlus, x, plus);
        double[] dwMinus = lambdaTimes(oMinus, vMinus, x, minus);
        if (omegas != null) {
            // d dist / d Omega = 2 (Omega (x - w)) (x - w)^T
            double mlr = lr * 0.1;
            double[] diffPlus = new double[d];
            double[] diffMinus = new double[d];
            for (int j = 0; j < d; j++) {
                diffPlus[j] = x[j] - prototypes[plus][j];
                diffMinus[j] = x[j] - prototypes[minus][j];
            }
            if (oPlus == oMinus) {
                for (int a = 0; a < oPlus.length; a++) {
                    for (int j = 0; j < d; j++) {
                        oPlus[a][j] -= mlr * 2 * (gPlus * vPlus[a] * diffPlus[j] + gMinus * vMinus[a] * diffMinus[j]);
                    }
                }
                normalise(oPlus);
            } else {
                for (int a = 0; a < oPlus.length; a++) {
                    for (int j = 0; j < d; j++) {
                        oPlus[a][j] -= mlr * 2 * gPlus * vPlus[a] * diffPlus[j];
                        oMinus[a][j] -= mlr * 2 * gMinus * vMinus[a] * diffMinus[j];
                    }
                }
                normalise(oPlus);
                normalise(oMinus);
            }
        }
        for (int j = 0; j < d; j++) {
            prototypes[plus][j] += lr * 2 * gPlus * dwPlus[j];
            prototypes[minus][j] += lr * 2 * gMinus * dwMinus[j];
        }
    }

    /** {@code Omega^T v} with {@code v = Omega (x - w)}, or {@code x - w} without a metric. */
    private double[] lambdaTimes(double[][] omega, double[] v, double[] x, int prototype) {
        int d = x.length;
        double[] out = new double[d];
        if (omega == null) {
            for (int j = 0; j < d; j++) {
                out[j] = x[j] - prototypes[prototype][j];
            }
            return out;
        }
        for (int a = 0; a < omega.length; a++) {
            for (int j = 0; j < d; j++) {
                out[j] += omega[a][j] * v[a];
            }
        }
        return out;
    }

    /** Scales {@code Omega} so that {@code trace(Omega^T Omega) = 1}. */
    private static void normalise(double[][] omega) {
        double s = 0;
        for (double[] row : omega) {
            for (double v : row) {
                s += v * v;
            }
        }
        double f = 1 / Math.sqrt(Math.max(1e-300, s));
        for (double[] row : omega) {
            for (int j = 0; j < row.length; j++) {
                row[j] *= f;
            }
        }
    }

    /**
     * Predicts a class.
     *
     * @param x a standardised point
     * @return the label of the nearest prototype under the learned metric
     */
    public int predict(double[] x) {
        int best = 0;
        double bd = Double.POSITIVE_INFINITY;
        for (int k = 0; k < prototypes.length; k++) {
            double dk = distance(x, k);
            if (dk < bd) {
                bd = dk;
                best = k;
            }
        }
        return labels[best];
    }

    /**
     * Predicts many points.
     *
     * @param x points
     * @return labels
     */
    public int[] predict(double[][] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = predict(x[i]);
        }
        return out;
    }

    /**
     * The relevance matrix {@code Lambda = Omega^T Omega}: the global one, or the average over
     * prototypes for the localised variant, or the identity divided by {@code d} without a metric.
     *
     * @return {@code [d][d]}
     */
    public double[][] relevanceMatrix() {
        int d = prototypes[0].length;
        double[][] out = new double[d][d];
        if (omegas == null) {
            for (int j = 0; j < d; j++) {
                out[j][j] = 1.0 / d;
            }
            return out;
        }
        for (double[][] o : omegas) {
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    double s = 0;
                    for (double[] row : o) {
                        s += row[a] * row[b];
                    }
                    out[a][b] += s / omegas.length;
                }
            }
        }
        return out;
    }

    /** @return the diagonal of {@link #relevanceMatrix()}: one relevance per feature, summing to 1 */
    public double[] relevanceProfile() {
        double[][] l = relevanceMatrix();
        double[] out = new double[l.length];
        for (int j = 0; j < l.length; j++) {
            out[j] = l[j][j];
        }
        return out;
    }

    /** @return the prototypes (a copy) */
    public double[][] prototypes() {
        double[][] out = new double[prototypes.length][];
        for (int k = 0; k < out.length; k++) {
            out[k] = prototypes[k].clone();
        }
        return out;
    }

    /** @return the prototypes' labels */
    public int[] prototypeLabels() {
        return labels.clone();
    }

    /** @return the rule */
    public Kind kind() {
        return kind;
    }
}
