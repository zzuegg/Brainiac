package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.PersistentHomology;
import io.github.zzuegg.brainiac.experimental.nn.SetEncoder;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Classifying point clouds by their topology with persistent homology.
 *
 * <p>Five classes of noisy 3-D clouds with 60 to 90 points: a Gaussian blob, a circle, two
 * disjoint circles, a figure-eight and a torus, each scaled to unit RMS radius. Models train on
 * clouds in a canonical pose and are tested in that pose and under random 3-D rotations.
 * Compared: an MLP on Rips barcode features ({@link PersistentHomology}: the {@code H1}
 * persistence image, the {@code H0} and {@code H1} Betti curves and the largest {@code H0}
 * deaths), an MLP on the raw (centred, truncated) coordinates, and a Deep Sets
 * {@link SetEncoder}. Also reported: barcode compute time against the number of points.
 */
public final class TopologicalFeatures {

    private static final int CLASSES = 5;
    private static final int MIN_POINTS = 60;
    private static final double MAX_RADIUS = 1.2;

    private TopologicalFeatures() {
    }

    private static double[] circle(SplittableRandom r, double cx, double radius) {
        double a = 2 * Math.PI * r.nextDouble();
        return new double[] {cx + radius * Math.cos(a), radius * Math.sin(a), 0};
    }

    static double[][] cloud(int label, int n, double noise, SplittableRandom r) {
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            x[i] = switch (label) {
                case 0 -> new double[] {r.nextGaussian(), r.nextGaussian(), 0.6 * r.nextGaussian()};
                case 1 -> circle(r, 0, 1);
                case 2 -> circle(r, r.nextBoolean() ? -1.4 : 1.4, 0.8);
                case 3 -> circle(r, r.nextBoolean() ? -1 : 1, 1);
                default -> {
                    double u = 2 * Math.PI * r.nextDouble();
                    double v = 2 * Math.PI * r.nextDouble();
                    double rad = 2 + 0.8 * Math.cos(v);
                    yield new double[] {rad * Math.cos(u), rad * Math.sin(u), 0.8 * Math.sin(v)};
                }
            };
        }
        double[] mean = new double[3];
        for (double[] p : x) {
            for (int k = 0; k < 3; k++) {
                mean[k] += p[k] / n;
            }
        }
        double rms = 0;
        for (double[] p : x) {
            for (int k = 0; k < 3; k++) {
                p[k] -= mean[k];
                rms += p[k] * p[k] / n;
            }
        }
        rms = Math.sqrt(rms);
        for (double[] p : x) {
            for (int k = 0; k < 3; k++) {
                p[k] = p[k] / rms + noise * r.nextGaussian();
            }
        }
        return x;
    }

    private static double[][] rotate(double[][] x, SplittableRandom r) {
        double[][] q = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 3; k++) {
                q[i][k] = r.nextGaussian();
            }
            for (int j = 0; j < i; j++) {
                double dot = q[i][0] * q[j][0] + q[i][1] * q[j][1] + q[i][2] * q[j][2];
                for (int k = 0; k < 3; k++) {
                    q[i][k] -= dot * q[j][k];
                }
            }
            double norm = Math.sqrt(q[i][0] * q[i][0] + q[i][1] * q[i][1] + q[i][2] * q[i][2]);
            for (int k = 0; k < 3; k++) {
                q[i][k] /= norm;
            }
        }
        double[][] out = new double[x.length][3];
        for (int i = 0; i < x.length; i++) {
            for (int k = 0; k < 3; k++) {
                for (int j = 0; j < 3; j++) {
                    out[i][k] += x[i][j] * q[j][k];
                }
            }
        }
        return out;
    }

    /**
     * Barcode features of one cloud.
     *
     * @param x the points
     * @return the H1 persistence image (6 x 6), Betti curves (12 bins each) and the five largest
     *     H0 deaths
     */
    static double[] topologicalFeatures(double[][] x) {
        PersistentHomology.Barcode b = PersistentHomology.rips(x, MAX_RADIUS);
        List<PersistentHomology.Pair> h1 = PersistentHomology.cap(b.h1(), MAX_RADIUS);
        double[] img = PersistentHomology.persistenceImage(h1, 6, MAX_RADIUS, MAX_RADIUS, 0.1);
        double[] b0 = PersistentHomology.bettiCurve(b.h0(), MAX_RADIUS, 12);
        double[] b1 = PersistentHomology.bettiCurve(h1, MAX_RADIUS, 12);
        double[] deaths = b.h0().stream().mapToDouble(PersistentHomology.Pair::death).sorted().toArray();
        double[] f = new double[img.length + 24 + 5];
        System.arraycopy(img, 0, f, 0, img.length);
        for (int k = 0; k < 12; k++) {
            f[img.length + k] = Math.log1p(b0[k]);
            f[img.length + 12 + k] = b1[k];
        }
        for (int k = 0; k < 5; k++) {
            f[img.length + 24 + k] = deaths[deaths.length - 1 - k];
        }
        return f;
    }

    private record Data(List<double[][]> clouds, int[] labels) {
    }

    private static Data data(int perClass, boolean rotated, SplittableRandom r) {
        List<double[][]> clouds = new ArrayList<>();
        int[] labels = new int[perClass * CLASSES];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = i % CLASSES;
            double[][] x = cloud(labels[i], MIN_POINTS + r.nextInt(31), 0.05, r);
            clouds.add(rotated ? rotate(x, r) : x);
        }
        return new Data(clouds, labels);
    }

    private static Tensor matrix(double[][] rows) {
        int d = rows[0].length;
        float[] v = new float[rows.length * d];
        for (int i = 0; i < rows.length; i++) {
            for (int k = 0; k < d; k++) {
                v[i * d + k] = (float) rows[i][k];
            }
        }
        return Tensor.of(Shape.of(rows.length, d), v);
    }

    private static Tensor oneHot(int[] labels) {
        float[] v = new float[labels.length * CLASSES];
        for (int i = 0; i < labels.length; i++) {
            v[i * CLASSES + labels[i]] = 1;
        }
        return Tensor.of(Shape.of(labels.length, CLASSES), v);
    }

    private static double accuracy(Tensor logits, int[] labels) {
        float[] p = logits.toArray();
        int ok = 0;
        for (int i = 0; i < labels.length; i++) {
            int best = 0;
            for (int c = 1; c < CLASSES; c++) {
                best = p[i * CLASSES + c] > p[i * CLASSES + best] ? c : best;
            }
            ok += best == labels[i] ? 1 : 0;
        }
        return (double) ok / labels.length;
    }

    private static Var crossEntropy(Var logits, Tensor y) {
        return logits.logsumexp(1, false).sub(logits.mul(y).sum(1, false)).mean();
    }

    private static double[][] featureRows(Data d, Function<double[][], double[]> features) {
        double[][] f = new double[d.labels().length][];
        for (int i = 0; i < f.length; i++) {
            f[i] = features.apply(d.clouds().get(i));
        }
        return f;
    }

    private static double[] mlpAccuracy(double[][] train, int[] yTrain, double[][][] tests, int[][] yTests,
            int steps, long seed) {
        int d = train[0].length;
        double[] mean = new double[d];
        double[] sd = new double[d];
        for (double[] row : train) {
            for (int k = 0; k < d; k++) {
                mean[k] += row[k] / train.length;
            }
        }
        for (double[] row : train) {
            for (int k = 0; k < d; k++) {
                sd[k] += Math.pow(row[k] - mean[k], 2) / train.length;
            }
        }
        Function<double[][], Tensor> standardise = rows -> {
            double[][] z = new double[rows.length][d];
            for (int i = 0; i < rows.length; i++) {
                for (int k = 0; k < d; k++) {
                    z[i][k] = (rows[i][k] - mean[k]) / Math.sqrt(sd[k] + 1e-6);
                }
            }
            return matrix(z);
        };
        Model m = Model.sequential().input(d).seed(seed).dense(64, Activation.RELU).dense(CLASSES, Activation.IDENTITY)
                .build();
        Optimizer opt = Optimizer.adam(3e-3f);
        Tensor x = standardise.apply(train);
        Tensor y = oneHot(yTrain);
        for (int s = 0; s < steps; s++) {
            crossEntropy(m.forward(Var.of(x), true), y).backward();
            opt.step(m.parameters());
        }
        double[] out = new double[tests.length];
        for (int t = 0; t < tests.length; t++) {
            out[t] = accuracy(m.forward(Var.of(standardise.apply(tests[t])), false).value(), yTests[t]);
        }
        return out;
    }

    private static double[] raw(double[][] x) {
        double[] f = new double[MIN_POINTS * 3];
        for (int i = 0; i < MIN_POINTS; i++) {
            System.arraycopy(x[i], 0, f, 3 * i, 3);
        }
        return f;
    }

    private static Tensor sets(Data d) {
        float[] v = new float[d.labels().length * MIN_POINTS * 3];
        for (int i = 0; i < d.labels().length; i++) {
            double[][] x = d.clouds().get(i);
            for (int p = 0; p < MIN_POINTS; p++) {
                for (int k = 0; k < 3; k++) {
                    v[(i * MIN_POINTS + p) * 3 + k] = (float) x[p][k];
                }
            }
        }
        return Tensor.of(Shape.of(d.labels().length, MIN_POINTS, 3), v);
    }

    /**
     * Trains on canonical poses and tests in the canonical pose and under random rotations.
     *
     * @param perClass training clouds per class (the tests have half as many)
     * @param seed the seed
     * @return method to {@code {accuracy, rotated accuracy}}
     */
    public static Map<String, double[]> classify(int perClass, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Data train = data(perClass, false, r);
        Data test = data(perClass / 2, false, r);
        Data rotated = data(perClass / 2, true, r);
        int[][] ys = {test.labels(), rotated.labels()};
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("barcode features + MLP", mlpAccuracy(featureRows(train, TopologicalFeatures::topologicalFeatures),
                train.labels(), new double[][][] {featureRows(test, TopologicalFeatures::topologicalFeatures),
                    featureRows(rotated, TopologicalFeatures::topologicalFeatures)}, ys, 300, seed));
        out.put("raw-point MLP", mlpAccuracy(featureRows(train, TopologicalFeatures::raw), train.labels(),
                new double[][][] {featureRows(test, TopologicalFeatures::raw),
                    featureRows(rotated, TopologicalFeatures::raw)}, ys, 300, seed));
        SetEncoder set = new SetEncoder(3, 32, CLASSES, SetEncoder.Kind.DEEP_SETS, 8, 4, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(3e-3f);
        Tensor x = sets(train);
        Tensor y = oneHot(train.labels());
        for (int s = 0; s < 300; s++) {
            crossEntropy(set.forward(Var.of(x), true), y).backward();
            opt.step(set.parameters());
        }
        out.put("Deep Sets", new double[] {accuracy(set.forward(Var.of(sets(test)), false).value(), test.labels()),
            accuracy(set.forward(Var.of(sets(rotated)), false).value(), rotated.labels())});
        return out;
    }

    /**
     * Rips barcode time against the number of points (a torus sample).
     *
     * @param sizes the point counts
     * @param seed the seed
     * @return {@code "n=..."} to {@code {milliseconds, H1 pairs}}
     */
    public static Map<String, double[]> timing(int[] sizes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int n : sizes) {
            double[][] x = cloud(4, n, 0.02, r);
            long t = System.nanoTime();
            PersistentHomology.Barcode b = PersistentHomology.rips(x, MAX_RADIUS);
            out.put("n=" + n, new double[] {(System.nanoTime() - t) / 1e6, b.h1().size()});
        }
        return out;
    }
}
