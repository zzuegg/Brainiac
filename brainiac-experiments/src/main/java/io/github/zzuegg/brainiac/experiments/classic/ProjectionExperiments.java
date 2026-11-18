package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.ml.ComponentAnalysis;
import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.LinearSvm;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.NeighbourEmbedding;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Component analysis and neighbour embeddings against PCA.
 * <ul>
 *   <li>{@link #cocktailParty}: three {@link SeasonalSeries} signals with different periods
 *       (plus a square wave) mixed linearly; FastICA against PCA by the mean absolute
 *       correlation of each true source with its best-matching recovered component;</li>
 *   <li>{@link #kernelPcaCircles}: concentric circles; a linear SVM on three RBF kernel-PCA
 *       components against one on the two PCA components (test accuracy);</li>
 *   <li>{@link #embeddings}: 2-D maps of MNIST-like digits and {@link ShapeImages} pixels by
 *       PCA, t-SNE and UMAP, scored by trustworthiness (k = 10) and leave-one-out 5-NN accuracy in 2-D.</li>
 * </ul>
 */
public final class ProjectionExperiments {

    private ProjectionExperiments() {
    }

    /**
     * Source recovery from a linear mixture.
     *
     * @param length the number of time steps
     * @param seed the seed
     * @return "ICA" and "PCA" to the mean best absolute correlation with the true sources
     */
    public static Map<String, Double> cocktailParty(int length, long seed) {
        float[][] raw = {
            SeasonalSeries.generate(length, 24, 0, 1, 0.02, seed),
            SeasonalSeries.generate(length, 61, 0, 1, 0.02, seed + 1),
        };
        double[][] sources = new double[3][length];
        for (int t = 0; t < length; t++) {
            sources[0][t] = raw[0][t];
            sources[1][t] = raw[1][t];
            sources[2][t] = (t / 17) % 2 == 0 ? 1 : -1;   // a square wave, strongly sub-Gaussian
        }
        SplittableRandom r = new SplittableRandom(seed);
        double[][] mix = new double[3][3];
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                mix[a][b] = r.nextDouble(0.2, 1.0);
            }
        }
        double[][] x = new double[length][3];
        for (int t = 0; t < length; t++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    x[t][a] += mix[a][b] * sources[b][t];
                }
            }
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("ICA", recovery(sources, ComponentAnalysis.fastIca(x, 3, 200, seed).transform(x)));
        out.put("PCA", recovery(sources, ComponentAnalysis.pca(x, 3).transform(x)));
        return out;
    }

    private static double recovery(double[][] sources, double[][] estimate) {
        double total = 0;
        for (double[] s : sources) {
            double best = 0;
            for (int c = 0; c < estimate[0].length; c++) {
                double[] e = new double[estimate.length];
                for (int t = 0; t < e.length; t++) {
                    e[t] = estimate[t][c];
                }
                best = Math.max(best, Math.abs(correlation(s, e)));
            }
            total += best / sources.length;
        }
        return total;
    }

    private static double correlation(double[] a, double[] b) {
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < a.length; i++) {
            ma += a[i] / a.length;
            mb += b[i] / b.length;
        }
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < a.length; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return sab / Math.sqrt(saa * sbb + 1e-300);
    }

    /**
     * Linear separability of concentric circles after PCA and kernel PCA.
     *
     * @param seed the seed
     * @return "PCA-2" (all of the input's components) and "kernel PCA-3" to the test accuracy of a
     *     linear SVM; with an RBF kernel the two leading kernel components are angular modes and
     *     the third is the radial one that separates the rings
     */
    public static Map<String, Double> kernelPcaCircles(long seed) {
        ClassicMlShootout.Points train = ClassicMlShootout.circles(300, 0.05, 0.4, seed);
        ClassicMlShootout.Points test = ClassicMlShootout.circles(300, 0.05, 0.4, seed + 1);
        Map<String, Double> out = new LinkedHashMap<>();
        ComponentAnalysis pca = ComponentAnalysis.pca(train.x(), 2);
        out.put("PCA-2", linearAccuracy(pca.transform(train.x()), train.y(), pca.transform(test.x()), test.y(), seed));
        ComponentAnalysis.Kernel kpca = ComponentAnalysis.kernelPca(train.x(), KernelFunction.rbf(1), 3);
        out.put("kernel PCA-3", linearAccuracy(kpca.transform(train.x()), train.y(), kpca.transform(test.x()),
                test.y(), seed));
        return out;
    }

    /** A linear SVM on standardised features: its test accuracy. */
    private static double linearAccuracy(double[][] train, int[] y, double[][] test, int[] yt, long seed) {
        double[] mean = MlMatrices.columnMeans(train);
        double[] sd = new double[mean.length];
        for (double[] row : train) {
            for (int c = 0; c < sd.length; c++) {
                sd[c] += (row[c] - mean[c]) * (row[c] - mean[c]) / train.length;
            }
        }
        double[][][] sets = {train, test};
        double[][][] scaled = new double[2][][];
        for (int s = 0; s < 2; s++) {
            scaled[s] = new double[sets[s].length][mean.length];
            for (int i = 0; i < sets[s].length; i++) {
                for (int c = 0; c < mean.length; c++) {
                    scaled[s][i][c] = (sets[s][i][c] - mean[c]) / Math.sqrt(sd[c] + 1e-24);
                }
            }
        }
        LinearSvm svm = LinearSvm.fit(scaled[0], y, 2, 1, 50, seed);
        return MlMetrics.accuracy(svm.predict(scaled[1]), yt);
    }

    /**
     * 2-D embeddings of the MNIST-like {@link ToyTasks#digits} (10 classes of noisy, shifted
     * 8x8 prototypes) and of {@link ShapeImages} pixels.
     *
     * @param perClass digits per class
     * @param shapes the number of shape images
     * @param seed the seed
     * @return "data / method" to {trustworthiness, 5-NN accuracy}
     */
    public static Map<String, double[]> embeddings(int perClass, int shapes, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        ToyTasks.Data digits = ToyTasks.digits(10, perClass, 8, 0.1, seed);
        embedAll("digits", MlMatrices.toDouble(digits.x()), digits.y(), seed, out);
        Tensor[] data = new ShapeImages(10, 0.15f).generate(shapes, seed);
        int pixels = 100;
        float[] img = data[0].toArray();
        float[] ids = data[1].toArray();
        double[][] x = new double[shapes][pixels];
        int[] y = new int[shapes];
        for (int i = 0; i < shapes; i++) {
            for (int p = 0; p < pixels; p++) {
                x[i][p] = img[i * pixels + p];
            }
            y[i] = (int) ids[i];
        }
        embedAll("shapes", x, y, seed, out);
        return out;
    }

    private static void embedAll(String name, double[][] x, int[] y, long seed, Map<String, double[]> out) {
        out.put(name + " / input space", new double[] {1, MlMetrics.knnAccuracy(x, y, 5)});
        out.put(name + " / PCA-2", score(x, ComponentAnalysis.pca(x, 2).transform(x), y));
        out.put(name + " / t-SNE", score(x, NeighbourEmbedding.tsne(x,
                new NeighbourEmbedding.TsneSettings(2, 20, 400, 100, 12, seed)), y));
        out.put(name + " / UMAP", score(x, NeighbourEmbedding.umap(x, NeighbourEmbedding.UmapSettings.defaults(15)), y));
    }

    private static double[] score(double[][] x, double[][] e, int[] y) {
        return new double[] {MlMetrics.trustworthiness(x, e, 10), MlMetrics.knnAccuracy(e, y, 5)};
    }
}
