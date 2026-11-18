package io.github.zzuegg.brainiac.experimental.bio;

/**
 * A k-mer logistic regression: each sequence becomes its normalised k-mer count vector (all
 * {@code A^k} k-mers), and an L2-penalised logistic regression is fitted by full-batch Adam.
 * The standard alignment-free baseline for family detection and bound/unbound classification.
 */
public final class KmerClassifier {

    private final int k;
    private final int alphabet;
    private final double[] w;
    private double b;

    /**
     * Creates an untrained classifier.
     *
     * @param k the k-mer length
     * @param alphabet the alphabet size
     */
    public KmerClassifier(int k, int alphabet) {
        this.k = k;
        this.alphabet = alphabet;
        this.w = new double[(int) Math.pow(alphabet, k)];
    }

    /**
     * The normalised k-mer counts of a sequence.
     *
     * @param seq the sequence
     * @return features summing to 1
     */
    public double[] features(int[] seq) {
        double[] f = new double[w.length];
        int total = 0;
        for (int i = 0; i + k <= seq.length; i++) {
            int code = 0;
            for (int j = 0; j < k; j++) {
                code = code * alphabet + seq[i + j];
            }
            f[code]++;
            total++;
        }
        for (int j = 0; j < f.length; j++) {
            f[j] = f[j] * 10 / Math.max(1, total);
        }
        return f;
    }

    /**
     * Fits by full-batch Adam.
     *
     * @param seqs sequences
     * @param labels labels
     * @param epochs gradient steps
     * @param l2 the weight decay
     * @return this classifier
     */
    public KmerClassifier fit(int[][] seqs, boolean[] labels, int epochs, double l2) {
        double[][] x = new double[seqs.length][];
        for (int i = 0; i < seqs.length; i++) {
            x[i] = features(seqs[i]);
        }
        double[] m = new double[w.length + 1];
        double[] v = new double[w.length + 1];
        for (int e = 1; e <= epochs; e++) {
            double[] g = new double[w.length + 1];
            for (int i = 0; i < x.length; i++) {
                double p = 1 / (1 + Math.exp(-dot(x[i])));
                double r = (p - (labels[i] ? 1 : 0)) / x.length;
                for (int j = 0; j < w.length; j++) {
                    g[j] += r * x[i][j];
                }
                g[w.length] += r;
            }
            for (int j = 0; j <= w.length; j++) {
                double gj = g[j] + (j < w.length ? l2 * w[j] : 0);
                m[j] = 0.9 * m[j] + 0.1 * gj;
                v[j] = 0.999 * v[j] + 0.001 * gj * gj;
                double step = 0.05 * (m[j] / (1 - Math.pow(0.9, e))) / (Math.sqrt(v[j] / (1 - Math.pow(0.999, e))) + 1e-8);
                if (j < w.length) {
                    w[j] -= step;
                } else {
                    b -= step;
                }
            }
        }
        return this;
    }

    private double dot(double[] f) {
        double s = b;
        for (int j = 0; j < f.length; j++) {
            s += w[j] * f[j];
        }
        return s;
    }

    /**
     * The logit of a sequence.
     *
     * @param seq the sequence
     * @return the score
     */
    public double score(int[] seq) {
        return dot(features(seq));
    }
}
