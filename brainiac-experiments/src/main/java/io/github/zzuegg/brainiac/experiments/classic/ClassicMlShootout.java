package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.ml.AffinityPropagation;
import io.github.zzuegg.brainiac.experimental.ml.AgglomerativeClustering;
import io.github.zzuegg.brainiac.experimental.ml.ClusteringMetrics;
import io.github.zzuegg.brainiac.experimental.ml.Dbscan;
import io.github.zzuegg.brainiac.experimental.ml.DensityClustering;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.KernelSvm;
import io.github.zzuegg.brainiac.experimental.ml.LinearSvm;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.NeighbourEmbedding;
import io.github.zzuegg.brainiac.experimental.ml.RandomFourierFeatures;
import io.github.zzuegg.brainiac.experimental.ml.SpectralClustering;
import io.github.zzuegg.brainiac.experimental.nn.MonotonicLayers;
import io.github.zzuegg.brainiac.experimental.nn.NeuralAdditiveModel;
import io.github.zzuegg.brainiac.experiments.generative.ToyPoints;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.function.UnaryOperator;

/**
 * The shared harness of the classic machine-learning experiments: seeded 2-D datasets (spirals
 * from {@link ToyTasks}, blobs from {@link ToyPoints}, concentric circles, two moons, anisotropic
 * and noisy blobs), the neural baselines (logistic regression and a small MLP built with core
 * {@code Model}), and two shootouts.
 * <ul>
 *   <li>{@link #kernelMachines}: an RBF kernel SVM (SMO), a polynomial SVM, random Fourier
 *       features with a linear SVM, and a Laplace GP classifier against logistic regression
 *       and an MLP: test accuracy, train time and support-vector count;</li>
 *   <li>{@link #clustering}: k-means with random and k-means++ seeding, a full-covariance GMM,
 *       spectral clustering and DBSCAN: the adjusted Rand index against the true labels, and
 *       for DBSCAN the recall of planted noise points;</li>
 *   <li>{@link #densityClustering} (wave 12): HDBSCAN, OPTICS, agglomerative clustering with four
 *       linkages and affinity propagation against k-means, a GMM, spectral clustering and DBSCAN
 *       with the best {@code eps} chosen on the labels, on blobs of different densities with
 *       noise, noisy moons, nested circles, anisotropic blobs and a UMAP embedding of five-class
 *       digits (400 points each);</li>
 *   <li>{@link #monotonic}: {@link MonotonicLayers} (min-max, lattice, constrained MLP) against
 *       an unconstrained MLP, a neural additive model, gradient-boosted trees and isotonic
 *       regression on a planted monotone target: RMSE in range, RMSE beyond the training range
 *       and the share of monotonicity violations.</li>
 * </ul>
 */
public final class ClassicMlShootout {

    private ClassicMlShootout() {
    }

    /**
     * A labelled point set.
     *
     * @param x the points {@code [n][d]}
     * @param y the labels; {@code -1} marks planted noise points
     * @param classes the number of real classes
     */
    public record Points(double[][] x, int[] y, int classes) {

        /**
         * A contiguous slice.
         *
         * @param from the first row
         * @param to one past the last row
         * @return the slice
         */
        public Points slice(int from, int to) {
            return new Points(java.util.Arrays.copyOfRange(x, from, to), java.util.Arrays.copyOfRange(y, from, to),
                    classes);
        }

        /** @return the number of points */
        public int size() {
            return x.length;
        }
    }

    /**
     * One shootout entry.
     *
     * @param score accuracy, ARI or another score (higher is better)
     * @param millis training time in milliseconds
     * @param size model size: support vectors, features or parameters (0 if not applicable)
     */
    public record Row(double score, double millis, int size) {
    }

    /**
     * Interleaved spirals from {@link ToyTasks#spiral}.
     *
     * @param perClass points per arm
     * @param classes arms
     * @param noise the angle noise
     * @param seed the seed
     * @return the points, shuffled
     */
    public static Points spirals(int perClass, int classes, double noise, long seed) {
        ToyTasks.Data d = ToyTasks.spiral(perClass, classes, noise, seed);
        return new Points(MlMatrices.toDouble(d.x()), d.y().clone(), classes);
    }

    /**
     * Two concentric noisy circles, the inner one {@code factor} times the outer radius.
     *
     * @param n the number of points
     * @param noise the Gaussian noise
     * @param factor the inner radius relative to the outer one
     * @param seed the seed
     * @return the points (label 0 outer, 1 inner)
     */
    public static Points circles(int n, double noise, double factor, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = i % 2;
            double a = r.nextDouble() * 2 * Math.PI;
            double rad = y[i] == 0 ? 1 : factor;
            x[i] = new double[] {rad * Math.cos(a) + noise * r.nextGaussian(), rad * Math.sin(a) + noise * r.nextGaussian()};
        }
        return new Points(x, y, 2);
    }

    /**
     * Two interleaving half circles.
     *
     * @param n the number of points
     * @param noise the Gaussian noise
     * @param seed the seed
     * @return the points
     */
    public static Points moons(int n, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = i % 2;
            double t = r.nextDouble() * Math.PI;
            double px = y[i] == 0 ? Math.cos(t) : 1 - Math.cos(t);
            double py = y[i] == 0 ? Math.sin(t) : 0.5 - Math.sin(t);
            x[i] = new double[] {px + noise * r.nextGaussian(), py + noise * r.nextGaussian()};
        }
        return new Points(x, y, 2);
    }

    /**
     * Gaussian blobs on a ring, from {@link ToyPoints#ring}.
     *
     * @param n the number of points
     * @param modes the number of blobs
     * @param spread the standard deviation of each blob
     * @param seed the seed
     * @return the points
     */
    public static Points blobs(int n, int modes, double spread, long seed) {
        ToyPoints toy = ToyPoints.ring(modes, 2.0, spread);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = r.nextInt(modes);
            float[] p = toy.sample(y[i], r);
            x[i] = new double[] {p[0], p[1]};
        }
        return new Points(x, y, modes);
    }

    /**
     * Blobs stretched by a shared linear map (the scikit-learn anisotropic example), which
     * spherical k-means splits wrongly and a full-covariance GMM fits.
     *
     * @param n the number of points
     * @param seed the seed
     * @return the points
     */
    public static Points anisotropic(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] centres = {{-2.5, 0}, {0, 0}, {2.5, 0}};
        double[][] x = new double[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = i % 3;
            double a = centres[y[i]][0] + 0.18 * r.nextGaussian();
            double b = centres[y[i]][1] + 2.0 * r.nextGaussian();
            // shear so that the long axis of each blob leans across the others
            x[i] = new double[] {a + 0.6 * b, 0.6 * b};
        }
        return new Points(x, y, 3);
    }

    /**
     * Blobs plus uniformly scattered noise points labelled {@code -1}.
     *
     * @param n the number of blob points
     * @param noisePoints the number of noise points
     * @param seed the seed
     * @return the points
     */
    public static Points noisyBlobs(int n, int noisePoints, long seed) {
        Points b = blobs(n, 4, 0.25, seed);
        SplittableRandom r = new SplittableRandom(seed + 99);
        double[][] x = java.util.Arrays.copyOf(b.x(), n + noisePoints);
        int[] y = java.util.Arrays.copyOf(b.y(), n + noisePoints);
        for (int i = n; i < n + noisePoints; i++) {
            x[i] = new double[] {r.nextDouble(-3.5, 3.5), r.nextDouble(-3.5, 3.5)};
            y[i] = -1;
        }
        return new Points(x, y, 4);
    }

    /**
     * Trains a softmax classifier with full-batch Adam and cross-entropy.
     *
     * @param model logits for {@code classes} classes
     * @param train the training points
     * @param steps Adam steps
     * @param lr the learning rate
     */
    public static void trainClassifier(Model model, Points train, int steps, float lr) {
        Tensor x = Tensor.of(MlMatrices.toFloat(train.x()));
        float[] ids = new float[train.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = train.y()[i];
        }
        Tensor y = Tensor.of(Shape.of(ids.length), ids);
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            Var loss = Functional.crossEntropy(model.forward(Var.of(x), true), y);
            loss.backward();
            opt.step(model.parameters());
        }
    }

    /**
     * Predicts classes with a trained classifier.
     *
     * @param model the classifier
     * @param x points
     * @return the arg-max classes
     */
    public static int[] predictClasses(Model model, double[][] x) {
        float[][] p = model.predict(MlMatrices.toFloat(x));
        int[] out = new int[p.length];
        for (int i = 0; i < p.length; i++) {
            int best = 0;
            for (int c = 1; c < p[i].length; c++) {
                if (p[i][c] > p[i][best]) {
                    best = c;
                }
            }
            out[i] = best;
        }
        return out;
    }

    /**
     * Logistic regression and kernel machines on one train/test split.
     *
     * @param train the training points
     * @param test the test points
     * @param gamma the RBF inverse width
     * @param seed the seed
     * @return method to test accuracy, train time and size
     */
    public static Map<String, Row> kernelMachines(Points train, Points test, double gamma, long seed) {
        Map<String, Row> out = new LinkedHashMap<>();
        int d = train.x()[0].length;
        long t0 = System.nanoTime();
        Model logistic = Model.sequential().input(d).dense(train.classes()).seed(seed).build();
        trainClassifier(logistic, train, 300, 0.05f);
        out.put("logistic", new Row(MlMetrics.accuracy(predictClasses(logistic, test.x()), test.y()), ms(t0),
                (int) logistic.parameterCount()));
        t0 = System.nanoTime();
        Model mlp = Model.sequential().input(d).dense(64, Activation.TANH).dense(64, Activation.TANH)
                .dense(train.classes()).seed(seed).build();
        trainClassifier(mlp, train, 600, 0.01f);
        out.put("MLP", new Row(MlMetrics.accuracy(predictClasses(mlp, test.x()), test.y()), ms(t0),
                (int) mlp.parameterCount()));
        t0 = System.nanoTime();
        KernelSvm rbf = KernelSvm.fit(train.x(), train.y(), train.classes(), KernelFunction.rbf(gamma), 10, 1e-3);
        out.put("SVM-RBF", new Row(MlMetrics.accuracy(rbf.predict(test.x()), test.y()), ms(t0), rbf.supportVectors()));
        t0 = System.nanoTime();
        KernelSvm poly = KernelSvm.fit(train.x(), train.y(), train.classes(), KernelFunction.polynomial(3, 1, 1), 10,
                1e-3);
        out.put("SVM-poly3", new Row(MlMetrics.accuracy(poly.predict(test.x()), test.y()), ms(t0),
                poly.supportVectors()));
        t0 = System.nanoTime();
        RandomFourierFeatures rff = new RandomFourierFeatures(d, 300, gamma, new SplittableRandom(seed));
        LinearSvm lin = LinearSvm.fit(rff.transform(train.x()), train.y(), train.classes(), 10, 40, seed);
        out.put("RFF+linear", new Row(MlMetrics.accuracy(lin.predict(rff.transform(test.x())), test.y()), ms(t0),
                rff.features()));
        t0 = System.nanoTime();
        LinearSvm raw = LinearSvm.fit(train.x(), train.y(), train.classes(), 10, 40, seed);
        out.put("linear SVM", new Row(MlMetrics.accuracy(raw.predict(test.x()), test.y()), ms(t0), d));
        if (train.classes() == 2) {
            t0 = System.nanoTime();
            int[] binary = new int[train.size()];
            for (int i = 0; i < binary.length; i++) {
                binary[i] = train.y()[i] == 1 ? 1 : 0;
            }
            GaussianProcess.Classifier gp = GaussianProcess.classifier(train.x(), binary,
                    (a, b) -> 4 * Math.exp(-gamma * MlMatrices.squaredDistance(a, b)));
            double[] p = gp.predictProbability(test.x());
            int[] pred = new int[p.length];
            for (int i = 0; i < p.length; i++) {
                pred[i] = p[i] > 0.5 ? 1 : 0;
            }
            out.put("GP-Laplace", new Row(MlMetrics.accuracy(pred, test.y()), ms(t0), train.size()));
        }
        return out;
    }

    /**
     * Clusterers against the true labels.
     *
     * @param points the points (noise points labelled {@code -1} count as their own group)
     * @param k the number of clusters given to the methods that need it
     * @param dbscanEps DBSCAN's radius
     * @param dbscanMin DBSCAN's minimum neighbourhood
     * @param seed the seed
     * @return method to adjusted Rand index (and train time); "DBSCAN noise recall" holds the
     *     share of planted noise points DBSCAN marked as noise, if there are any
     */
    public static Map<String, Row> clustering(Points points, int k, double dbscanEps, int dbscanMin, long seed) {
        Map<String, Row> out = new LinkedHashMap<>();
        double[][] x = points.x();
        int[] y = points.y();
        long t0 = System.nanoTime();
        KMeans random = KMeans.fit(x, k, KMeans.Init.RANDOM, 1, 100, seed);
        out.put("k-means random", new Row(MlMetrics.adjustedRandIndex(y, random.labels()), ms(t0), k));
        t0 = System.nanoTime();
        KMeans plus = KMeans.fit(x, k, KMeans.Init.PLUS_PLUS, 1, 100, seed);
        out.put("k-means++", new Row(MlMetrics.adjustedRandIndex(y, plus.labels()), ms(t0), k));
        t0 = System.nanoTime();
        GaussianMixture gmm = GaussianMixture.fit(x, k, 100, 1e-6, seed);
        out.put("GMM", new Row(MlMetrics.adjustedRandIndex(y, gmm.predict(x)), ms(t0), k));
        t0 = System.nanoTime();
        int[] spectral = SpectralClustering.cluster(SpectralClustering.knnAffinity(x, 10), k, seed);
        out.put("spectral", new Row(MlMetrics.adjustedRandIndex(y, spectral), ms(t0), k));
        t0 = System.nanoTime();
        int[] db = Dbscan.cluster(x, dbscanEps, dbscanMin);
        out.put("DBSCAN", new Row(MlMetrics.adjustedRandIndex(y, db), ms(t0), 0));
        int noise = 0;
        int caught = 0;
        for (int i = 0; i < y.length; i++) {
            if (y[i] < 0) {
                noise++;
                caught += db[i] == Dbscan.NOISE ? 1 : 0;
            }
        }
        if (noise > 0) {
            out.put("DBSCAN noise recall", new Row(caught / (double) noise, 0, noise));
        }
        return out;
    }

    private static Points densityData(String name, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        switch (name) {
            case "varying densities" -> {
                double[][] centres = {{0, 0}, {2.2, 0}, {1, 2.6}};
                double[] spread = {0.1, 0.25, 0.5};
                int n = 400;
                double[][] x = new double[n][];
                int[] y = new int[n];
                for (int i = 0; i < n; i++) {
                    if (i < 360) {
                        int c = i % 3;
                        y[i] = c;
                        x[i] = new double[] {centres[c][0] + spread[c] * r.nextGaussian(),
                            centres[c][1] + spread[c] * r.nextGaussian()};
                    } else {
                        y[i] = -1;
                        x[i] = new double[] {-1.5 + 5 * r.nextDouble(), -1.5 + 5.5 * r.nextDouble()};
                    }
                }
                return new Points(x, y, 3);
            }
            case "moons + noise" -> {
                Points m = moons(360, 0.06, seed);
                double[][] x = java.util.Arrays.copyOf(m.x(), 400);
                int[] y = java.util.Arrays.copyOf(m.y(), 400);
                for (int i = 360; i < 400; i++) {
                    x[i] = new double[] {-1.5 + 4 * r.nextDouble(), -1 + 2.5 * r.nextDouble()};
                    y[i] = -1;
                }
                return new Points(x, y, 2);
            }
            case "circles" -> {
                return circles(400, 0.05, 0.5, seed);
            }
            case "anisotropic" -> {
                return anisotropic(400, seed);
            }
            default -> {
                ToyTasks.Data d = ToyTasks.digits(5, 80, 8, 0.05, seed);
                double[][] e = NeighbourEmbedding.umap(MlMatrices.toDouble(d.x()),
                        NeighbourEmbedding.UmapSettings.defaults(15));
                return new Points(e, d.y().clone(), 5);
            }
        }
    }

    private static double[] scores(Points p, int[] labels, boolean hasNoise) {
        int[] y = p.y();
        int clusters = (int) java.util.Arrays.stream(labels).filter(v -> v >= 0).distinct().count();
        double precision = Double.NaN;
        if (hasNoise) {
            int called = 0;
            int right = 0;
            for (int i = 0; i < y.length; i++) {
                if (labels[i] < 0) {
                    called++;
                    right += y[i] < 0 ? 1 : 0;
                }
            }
            precision = called > 0 ? (double) right / called : Double.NaN;
        }
        // noise (-1) becomes its own label 0 for the metrics
        int[] ty = java.util.Arrays.stream(y).map(v -> v + 1).toArray();
        int[] tl = java.util.Arrays.stream(labels).map(v -> v + 1).toArray();
        return new double[] {MlMetrics.adjustedRandIndex(ty, tl), ClusteringMetrics.nmi(ty, tl), clusters, precision};
    }

    /**
     * Density-based and exemplar clustering (wave 12, item 12.5.18). HDBSCAN uses
     * {@code minSamples = 5} and {@code minClusterSize} 15 (with 5 and 30 as a sensitivity check);
     * DBSCAN takes the {@code eps} (with {@code minPts = 5}) that maximises the ARI on the labels,
     * an optimistic baseline; OPTICS is cut at twice its median reachability; k-means, the GMM,
     * spectral and agglomerative clustering get the true number of clusters; affinity propagation
     * uses the median similarity as preference.
     *
     * @param datasets data sets from {@code "varying densities"}, {@code "moons + noise"},
     *     {@code "circles"}, {@code "anisotropic"} and {@code "digits UMAP"}
     * @param seed the seed
     * @return {@code "data / method"} to {ARI, NMI, clusters found, precision of the points called
     *     noise (NaN when none are called or the data has no planted noise)}
     */
    public static Map<String, double[]> densityClustering(String[] datasets, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String name : datasets) {
            Points p = densityData(name, seed);
            double[][] x = p.x();
            int k = p.classes();
            boolean noise = java.util.Arrays.stream(p.y()).anyMatch(v -> v < 0);
            String key = name + " / ";
            out.put(key + "k-means++", scores(p, KMeans.fit(x, k, KMeans.Init.PLUS_PLUS, 3, 100, seed).labels(),
                    noise));
            out.put(key + "GMM", scores(p, GaussianMixture.fit(x, k, 100, 1e-6, seed).predict(x), noise));
            out.put(key + "spectral", scores(p, SpectralClustering.cluster(SpectralClustering.knnAffinity(x, 10), k,
                    seed), noise));
            double[] kth = new double[x.length];
            double[][] d2 = MlMatrices.squaredDistances(x);
            for (int i = 0; i < x.length; i++) {
                int[] nn = MlMetrics.nearest(d2[i], i, 5);
                kth[i] = Math.sqrt(d2[i][nn[4]]);
            }
            double[] sorted = kth.clone();
            java.util.Arrays.sort(sorted);
            double median = sorted[sorted.length / 2];
            double[] bestDb = null;
            for (int g = 0; g < 24; g++) {
                double eps = median * (0.5 + g * 0.15);
                double[] sc = scores(p, Dbscan.cluster(x, eps, 5), noise);
                if (bestDb == null || sc[0] > bestDb[0]) {
                    bestDb = sc;
                }
            }
            out.put(key + "DBSCAN (oracle eps)", bestDb);
            DensityClustering.Optics optics = DensityClustering.optics(x, 5);
            double[] reach = java.util.Arrays.stream(optics.reachability()).filter(Double::isFinite).sorted().toArray();
            out.put(key + "OPTICS (cut 2x median reach)",
                    scores(p, optics.extractDbscan(2 * reach[reach.length / 2]), noise));
            for (int mcs : new int[] {15, 5, 30}) {
                out.put(key + "HDBSCAN mcs=" + mcs, scores(p, DensityClustering.hdbscan(x, mcs, 5).labels(), noise));
            }
            for (AgglomerativeClustering.Linkage l : AgglomerativeClustering.Linkage.values()) {
                out.put(key + "agglomerative " + l, scores(p, AgglomerativeClustering.cluster(x, k, l), noise));
            }
            out.put(key + "affinity propagation", scores(p, AffinityPropagation.cluster(x, Double.NaN, 0.7, 200, 15)
                    .labels(), noise));
        }
        return out;
    }

    private static double ms(long t0) {
        return (System.nanoTime() - t0) / 1e6;
    }

    /**
     * One monotonic-regression entry.
     *
     * @param rmse test RMSE inside the training range
     * @param extrapolationRmse test RMSE where the first feature exceeds the training range
     * @param violations share of grid pairs where the output drops along a monotone feature
     */
    public record MonotoneRow(double rmse, double extrapolationRmse, double violations) {
    }

    /** The planted monotone shape of feature {@code j}. */
    private static double plantedMonotone(int j, double x) {
        return switch (j % 4) {
            case 0 -> 0.8 * x;
            case 1 -> Math.tanh(2 * x);
            case 2 -> 0.15 * x * x * x;
            default -> x > 0 ? 0.7 : -0.3 + 0.2 * x;
        };
    }

    /**
     * Monotonic networks against unconstrained models (item 11.4.17). Features are the 8 columns
     * of a {@link TabularStressSuite} table (uniform in
     * {@code [-2, 2]}); the target is {@code Σ_{j < k} m_j(x_j)} with increasing planted shapes,
     * plus {@code 0.5 sin(2 x_k)} when {@code k > 1}, plus noise 0.1. Training and in-range test
     * rows have {@code x_0 <= 1}; the extrapolation set has {@code x_0 > 1}.
     *
     * @param rows training rows (100 to 500)
     * @param monotoneFeatures {@code k}, 1 to 4
     * @param steps Adam steps for the neural models
     * @param seed the seed
     * @return MLP, min-max, lattice, constrained MLP, NAM, GBT, and isotonic regression on
     *     {@code x_0} when {@code k = 1}
     */
    public static Map<String, MonotoneRow> monotonic(int rows, int monotoneFeatures, int steps, long seed) {
        var suite = new TabularStressSuite(4, 0.5, false, seed);
        int d = suite.columns();
        List<double[]> inRange = new ArrayList<>();
        List<double[]> beyond = new ArrayList<>();
        float[] raw = suite.table(rows * 4 + 2000, seed + 1).features().toArray();
        for (int i = 0; i < raw.length / d; i++) {
            double[] x = new double[d];
            for (int j = 0; j < d; j++) {
                x[j] = raw[i * d + j];
            }
            (x[0] <= 1 ? inRange : beyond).add(x);
        }
        SplittableRandom r = new SplittableRandom(seed + 2);
        ToDoubleFunction<double[]> target = x -> {
            double s = 0;
            for (int j = 0; j < monotoneFeatures; j++) {
                s += plantedMonotone(j, x[j]);
            }
            return monotoneFeatures > 1 ? s + 0.5 * Math.sin(2 * x[monotoneFeatures]) : s;
        };
        double[][] xtr = inRange.subList(0, rows).toArray(new double[0][]);
        double[][] xte = inRange.subList(rows, Math.min(inRange.size(), rows + 1000)).toArray(new double[0][]);
        double[][] xex = beyond.subList(0, Math.min(beyond.size(), 500)).toArray(new double[0][]);
        double[] ytr = new double[xtr.length];
        for (int i = 0; i < ytr.length; i++) {
            ytr[i] = target.applyAsDouble(xtr[i]) + 0.1 * r.nextGaussian();
        }
        boolean[] monotone = new boolean[d];
        double[] low = new double[d];
        double[] high = new double[d];
        for (int j = 0; j < d; j++) {
            monotone[j] = j < monotoneFeatures;
            low[j] = -2;
            high[j] = 2;
        }
        Map<String, Function<double[][], double[]>> models = new LinkedHashMap<>();
        Model mlp = Model.sequential().input(d).dense(32, Activation.TANH).dense(32, Activation.TANH).dense(1)
                .seed(seed).build();
        models.put("MLP", regressor(v -> mlp.forward(v, true), mlp.parameters(), xtr, ytr, steps));
        for (MonotonicLayers.Kind k
                : MonotonicLayers.Kind.values()) {
            Layer layer = MonotonicLayers.create(k, monotone, low, high, seed);
            models.put(k.name(), regressor(layer::forward, layer.parameters(), xtr, ytr, steps));
        }
        NeuralAdditiveModel nam = new NeuralAdditiveModel(d, 16, new SplittableRandom(seed));
        models.put("NAM", regressor(nam::forward, nam.parameters(), xtr, ytr, steps));
        var gbt = GradientBoostedTrees.fit(xtr, ytr,
                GradientBoostedTrees.Objective.REGRESSION,
                GradientBoostedTrees.Settings.defaults().withDepth(3));
        models.put("GBT", gbt::predict);
        if (monotoneFeatures == 1) {
            double[] s = new double[xtr.length];
            for (int i = 0; i < s.length; i++) {
                s[i] = xtr[i][0];
            }
            double[][] map = Recalibrators.isotonic(s, ytr);
            models.put("isotonic", x -> {
                double[] out = new double[x.length];
                for (int i = 0; i < x.length; i++) {
                    out[i] = interpolate(map, x[i][0]);
                }
                return out;
            });
        }
        Map<String, MonotoneRow> out = new LinkedHashMap<>();
        for (var e : models.entrySet()) {
            out.put(e.getKey(), new MonotoneRow(rmse(e.getValue().apply(xte), xte, target),
                    rmse(e.getValue().apply(xex), xex, target),
                    MonotonicLayers.violations(e.getValue(), low, high,
                            monotone, 21, 50, seed + 3)));
        }
        return out;
    }

    /** Piecewise-linear interpolation of an isotonic fit, flat beyond its ends. */
    private static double interpolate(double[][] map, double x) {
        double[] xs = map[0];
        double[] ys = map[1];
        if (x <= xs[0]) {
            return ys[0];
        }
        for (int b = 1; b < xs.length; b++) {
            if (x <= xs[b]) {
                double t = (x - xs[b - 1]) / (xs[b] - xs[b - 1]);
                return ys[b - 1] + t * (ys[b] - ys[b - 1]);
            }
        }
        return ys[ys.length - 1];
    }

    private static double rmse(double[] pred, double[][] x, ToDoubleFunction<double[]> target) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            double e = pred[i] - target.applyAsDouble(x[i]);
            s += e * e;
        }
        return Math.sqrt(s / x.length);
    }

    /** Full-batch Adam on squared error; returns the trained model as a function. */
    private static Function<double[][], double[]> regressor(UnaryOperator<Var> forward, List<Var> params,
            double[][] x, double[] y, int steps) {
        Tensor xt = Tensor.of(MlMatrices.toFloat(x));
        float[] yv = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            yv[i] = (float) y[i];
        }
        Tensor yt = Tensor.of(Shape.of(y.length, 1), yv);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            Var loss = forward.apply(Var.of(xt)).reshape(y.length, 1).sub(yt).square().mean();
            loss.backward();
            opt.step(params);
        }
        return rowsIn -> {
            float[] p = forward.apply(Var.of(Tensor.of(MlMatrices.toFloat(rowsIn)))).value().toArray();
            double[] out = new double[rowsIn.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = p[i];
            }
            return out;
        };
    }
}
