package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.Dbscan;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.KernelSvm;
import io.github.zzuegg.brainiac.experimental.ml.LinearSvm;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.RandomFourierFeatures;
import io.github.zzuegg.brainiac.experimental.ml.SpectralClustering;
import io.github.zzuegg.brainiac.experiments.generative.ToyPoints;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

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
 *       for DBSCAN the recall of planted noise points.</li>
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

    private static double ms(long t0) {
        return (System.nanoTime() - t0) / 1e6;
    }
}
