package io.github.zzuegg.brainiac.experimental.calibrate;

/**
 * Multinomial logistic regression of the labels on a feature row per sample (logits or
 * log-probabilities), {@code softmax(W f + b)}, with the structures and ridge penalties the
 * affine recalibration maps need, fitted by full-batch Adam with a cosine step decay.
 * Package-private numerics shared by {@link Recalibrators} and {@link StructuredMatrixScaling}.
 */
final class AffineLogitFit {

    /** How {@code W} is parametrised. */
    enum Structure {
        /** {@code W = t I}: temperature-like */
        SCALAR,
        /** {@code W = diag(w)} */
        DIAGONAL,
        /** a full matrix */
        FULL,
        /** {@code W = I + diag(v) + U V^T} */
        LOW_RANK
    }

    final double[][] w;
    final double[] b;

    private AffineLogitFit(double[][] w, double[] b) {
        this.w = w;
        this.b = b;
    }

    /**
     * Fits the map.
     *
     * @param f features {@code [n][k]}
     * @param y labels
     * @param structure the parametrisation
     * @param rank the rank of {@code U V^T} for {@link Structure#LOW_RANK}
     * @param diagPenalty ridge on the deviation of the diagonal from one
     * @param offPenalty ridge on the off-diagonal entries (or on {@code U, V})
     * @param biasPenalty ridge on the intercept
     * @param iterations Adam iterations
     * @return the fitted map
     */
    static AffineLogitFit fit(double[][] f, int[] y, Structure structure, int rank, double diagPenalty,
            double offPenalty, double biasPenalty, int iterations) {
        int n = f.length;
        int k = f[0].length;
        // parameter layout: W (k*k) or the low-rank factors, then b
        int r = structure == Structure.LOW_RANK ? rank : 0;
        double[][] w = identity(k);
        double[] b = new double[k];
        double[][] u = new double[k][r];
        double[][] v = new double[k][r];
        double[] diag = new double[k];
        java.util.SplittableRandom rnd = new java.util.SplittableRandom(17);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < r; j++) {
                u[i][j] = 0.01 * rnd.nextGaussian();
                v[i][j] = 0.01 * rnd.nextGaussian();
            }
        }
        int size = structure == Structure.LOW_RANK ? k + 2 * k * r + k : k * k + k;
        double[] m1 = new double[size];
        double[] m2 = new double[size];
        double[] grad = new double[size];
        double[][] gw = new double[k][k];
        double[] z = new double[k];
        for (int it = 1; it <= iterations; it++) {
            if (structure == Structure.LOW_RANK) {
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        double s = i == j ? 1 + diag[i] : 0;
                        for (int q = 0; q < r; q++) {
                            s += u[i][q] * v[j][q];
                        }
                        w[i][j] = s;
                    }
                }
            }
            for (double[] row : gw) {
                java.util.Arrays.fill(row, 0);
            }
            double[] gb = new double[k];
            for (int s = 0; s < n; s++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < k; i++) {
                    double a = b[i];
                    for (int j = 0; j < k; j++) {
                        a += w[i][j] * f[s][j];
                    }
                    z[i] = a;
                    max = Math.max(max, a);
                }
                double sum = 0;
                for (int i = 0; i < k; i++) {
                    z[i] = Math.exp(z[i] - max);
                    sum += z[i];
                }
                for (int i = 0; i < k; i++) {
                    double g = (z[i] / sum - (i == y[s] ? 1 : 0)) / n;
                    gb[i] += g;
                    for (int j = 0; j < k; j++) {
                        gw[i][j] += g * f[s][j];
                    }
                }
            }
            for (int i = 0; i < k; i++) {
                gb[i] += 2 * biasPenalty * b[i];
            }
            int p = 0;
            if (structure == Structure.LOW_RANK) {
                for (int i = 0; i < k; i++) {
                    grad[p++] = gw[i][i] + 2 * diagPenalty * diag[i];
                }
                for (int i = 0; i < k; i++) {
                    for (int q = 0; q < r; q++) {
                        double gu = 2 * offPenalty * u[i][q];
                        double gv = 2 * offPenalty * v[i][q];
                        for (int j = 0; j < k; j++) {
                            gu += gw[i][j] * v[j][q];
                            gv += gw[j][i] * u[j][q];
                        }
                        grad[p++] = gu;
                        grad[p++] = gv;
                    }
                }
            } else {
                double trace = 0;
                for (int i = 0; i < k; i++) {
                    trace += gw[i][i] + 2 * diagPenalty * (w[i][i] - 1);
                }
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        double g;
                        if (i == j) {
                            g = structure == Structure.SCALAR ? trace / k : gw[i][i] + 2 * diagPenalty * (w[i][i] - 1);
                        } else {
                            g = structure == Structure.FULL ? gw[i][j] + 2 * offPenalty * w[i][j] : 0;
                        }
                        grad[p++] = g;
                    }
                }
            }
            for (int i = 0; i < k; i++) {
                grad[p++] = gb[i];
            }
            double lr = 0.001 + 0.5 * (0.05 - 0.001) * (1 + Math.cos(Math.PI * it / iterations));
            double c1 = 1 - Math.pow(0.9, it);
            double c2 = 1 - Math.pow(0.999, it);
            double[] step = new double[size];
            for (int q = 0; q < size; q++) {
                m1[q] = 0.9 * m1[q] + 0.1 * grad[q];
                m2[q] = 0.999 * m2[q] + 0.001 * grad[q] * grad[q];
                step[q] = lr * (m1[q] / c1) / (Math.sqrt(m2[q] / c2) + 1e-8);
            }
            p = 0;
            if (structure == Structure.LOW_RANK) {
                for (int i = 0; i < k; i++) {
                    diag[i] -= step[p++];
                }
                for (int i = 0; i < k; i++) {
                    for (int q = 0; q < r; q++) {
                        u[i][q] -= step[p++];
                        v[i][q] -= step[p++];
                    }
                }
            } else {
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        w[i][j] -= step[p++];
                    }
                }
            }
            for (int i = 0; i < k; i++) {
                b[i] -= step[p++];
            }
        }
        if (structure == Structure.LOW_RANK) {
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    double s = i == j ? 1 + diag[i] : 0;
                    for (int q = 0; q < r; q++) {
                        s += u[i][q] * v[j][q];
                    }
                    w[i][j] = s;
                }
            }
        }
        return new AffineLogitFit(w, b);
    }

    /** Applies the map to feature rows and returns probabilities. */
    double[][] probabilities(double[][] f) {
        double[][] out = new double[f.length][];
        int k = b.length;
        for (int s = 0; s < f.length; s++) {
            double[] z = new double[k];
            for (int i = 0; i < k; i++) {
                double a = b[i];
                for (int j = 0; j < k; j++) {
                    a += w[i][j] * f[s][j];
                }
                z[i] = a;
            }
            out[s] = softmax(z);
        }
        return out;
    }

    static double[][] identity(int k) {
        double[][] w = new double[k][k];
        for (int i = 0; i < k; i++) {
            w[i][i] = 1;
        }
        return w;
    }

    static double[] softmax(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) {
            max = Math.max(max, v);
        }
        double[] p = new double[z.length];
        double s = 0;
        for (int i = 0; i < z.length; i++) {
            p[i] = Math.exp(z[i] - max);
            s += p[i];
        }
        for (int i = 0; i < z.length; i++) {
            p[i] /= s;
        }
        return p;
    }

    /** Log-softmax rows of logits divided by a temperature, clipped below at {@code floor}. */
    static double[][] logProbabilities(double[][] logits, double temperature, double floor) {
        double[][] out = new double[logits.length][];
        for (int s = 0; s < logits.length; s++) {
            double[] z = logits[s];
            double max = Double.NEGATIVE_INFINITY;
            for (double v : z) {
                max = Math.max(max, v / temperature);
            }
            double sum = 0;
            for (double v : z) {
                sum += Math.exp(v / temperature - max);
            }
            double lse = max + Math.log(sum);
            out[s] = new double[z.length];
            for (int i = 0; i < z.length; i++) {
                out[s][i] = Math.max(floor, z[i] / temperature - lse);
            }
        }
        return out;
    }

    /** Rows of probabilities from logits. */
    static double[][] softmaxRows(double[][] logits) {
        double[][] out = new double[logits.length][];
        for (int s = 0; s < logits.length; s++) {
            out[s] = softmax(logits[s]);
        }
        return out;
    }

    /** Renormalises rows of non-negative scores, flooring each entry at {@code floor}. */
    static double[][] renormalise(double[][] scores, double floor) {
        double[][] out = new double[scores.length][];
        for (int s = 0; s < scores.length; s++) {
            int k = scores[s].length;
            out[s] = new double[k];
            double sum = 0;
            for (int i = 0; i < k; i++) {
                out[s][i] = Math.max(floor, scores[s][i]);
                sum += out[s][i];
            }
            for (int i = 0; i < k; i++) {
                out[s][i] /= sum;
            }
        }
        return out;
    }
}
