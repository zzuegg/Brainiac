package io.github.zzuegg.brainiac.experiments.metric;

import io.github.zzuegg.brainiac.experimental.ml.ClusteringMetrics;
import io.github.zzuegg.brainiac.experimental.ml.DeepEmbeddedClustering;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.NeighbourClustering;
import io.github.zzuegg.brainiac.experimental.ml.SpectralClustering;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Wave 10 items 10.3.11-10.3.12: deep clustering of unlabelled toy digits
 * ({@link ToyTasks#digits}, 10 classes), scored against the true classes by Hungarian-matched
 * clustering accuracy, NMI and ARI ({@link ClusteringMetrics}).
 *
 * <p>Rows: k-means, a Gaussian mixture and spectral clustering on raw pixels; k-means on the
 * pretrained autoencoder's latents (DEC's initialisation); DEC and IDEC
 * ({@link DeepEmbeddedClustering}); IIC on raw pixels with shift-and-noise augmentation; and,
 * on a frozen SimCLR encoder pretrained on the same unlabelled digits through
 * {@link SelfSupervisedBenchmark#pretrainEncoder}, k-means, spectral clustering and SCAN
 * ({@link NeighbourClustering}).
 */
public final class DeepClustering {

    /**
     * Scores of one clustering.
     *
     * @param accuracy Hungarian-matched accuracy
     * @param nmi normalised mutual information
     * @param ari adjusted Rand index
     */
    public record Score(double accuracy, double nmi, double ari) {

        @Override
        public String toString() {
            return String.format("acc %.3f NMI %.3f ARI %.3f", accuracy, nmi, ari);
        }
    }

    private DeepClustering() {
    }

    /**
     * The digits to cluster.
     *
     * @param perClass images per class
     * @param seed the seed
     * @return the data (labels only for scoring)
     */
    public static ToyTasks.Data digits(int perClass, long seed) {
        return ToyTasks.digits(10, perClass, 8, 0.1, seed, seed * 31 + 7);
    }

    private static Score score(int[] truth, int[] labels) {
        return new Score(ClusteringMetrics.accuracy(truth, labels), ClusteringMetrics.nmi(truth, labels),
                ClusteringMetrics.ari(truth, labels));
    }

    /**
     * Item 10.3.11: the raw-pixel baselines, k-means on autoencoder latents, DEC and IDEC.
     *
     * @param perClass images per class
     * @param settings DEC's settings
     * @param seed the seed
     * @return row name to score, in order
     */
    public static Map<String, Score> embedded(int perClass, DeepEmbeddedClustering.Settings settings, long seed) {
        ToyTasks.Data d = digits(perClass, seed);
        double[][] x = MlMatrices.toDouble(d.x());
        Map<String, Score> out = new LinkedHashMap<>();
        out.put("k-means raw", score(d.y(), KMeans.fit(x, 10, KMeans.Init.PLUS_PLUS, 10, 100, seed).labels()));
        out.put("GMM raw", score(d.y(), GaussianMixture.fit(x, 10, 50, 1e-2, seed).predict(x)));
        out.put("spectral raw", score(d.y(), SpectralClustering.cluster(SpectralClustering.knnAffinity(x, 10), 10,
                seed)));
        DeepEmbeddedClustering.Result dec = DeepEmbeddedClustering.fit(d.x(), 10, DeepEmbeddedClustering.Mode.DEC,
                settings, seed);
        DeepEmbeddedClustering.Result idec = DeepEmbeddedClustering.fit(d.x(), 10, DeepEmbeddedClustering.Mode.IDEC,
                settings, seed);
        out.put("k-means AE latents", score(d.y(), dec.initialLabels()));
        out.put("DEC", score(d.y(), dec.labels()));
        out.put("IDEC", score(d.y(), idec.labels()));
        return out;
    }

    /** Shifts each image by up to one pixel (circularly) and adds noise. */
    private static Tensor augment(Tensor x, SplittableRandom r) {
        int n = x.shape().dim(0);
        float[] v = x.toArray();
        float[] out = new float[v.length];
        for (int i = 0; i < n; i++) {
            int dx = r.nextInt(3) - 1;
            int dy = r.nextInt(3) - 1;
            for (int py = 0; py < 8; py++) {
                for (int px = 0; px < 8; px++) {
                    out[i * 64 + py * 8 + px] = v[i * 64 + Math.floorMod(py - dy, 8) * 8 + Math.floorMod(px - dx, 8)]
                            + (float) (r.nextGaussian() * 0.1);
                }
            }
        }
        return Tensor.of(x.shape(), out);
    }

    /**
     * Item 10.3.12: IIC on raw pixels, and k-means, spectral clustering and SCAN on frozen
     * SimCLR features of the same digits.
     *
     * @param perClass images per class
     * @param pretrainSteps SimCLR steps
     * @param iicSteps IIC steps
     * @param scan SCAN's settings
     * @param seed the seed
     * @return row name to score, in order; also "SCAN neighbour purity" as accuracy
     */
    public static Map<String, Score> neighbourBased(int perClass, int pretrainSteps, int iicSteps,
            NeighbourClustering.ScanSettings scan, long seed) {
        ToyTasks.Data d = digits(perClass, seed);
        Map<String, Score> out = new LinkedHashMap<>();
        int[] iic = NeighbourClustering.iic(d.x(), 10, DeepClustering::augment,
                new NeighbourClustering.IicSettings(128, 50, iicSteps, 128), seed);
        out.put("IIC raw", score(d.y(), iic));
        // SimCLR pretrained on unlabelled digit batches (new samples of the same classes)
        UnaryOperator<Tensor> encoder = SelfSupervisedBenchmark.pretrainEncoder(SelfSupervisedBenchmark.Method.SIMCLR,
                s -> images(ToyTasks.digits(10, 7, 8, 0.1, seed, s).x()), pretrainSteps, seed);
        float[][] features = NeighbourClustering.rows(encoder.apply(images(d.x())));
        double[][] f = unit(features);
        out.put("k-means SimCLR", score(d.y(), KMeans.fit(f, 10, KMeans.Init.PLUS_PLUS, 10, 100, seed).labels()));
        out.put("spectral SimCLR", score(d.y(), SpectralClustering.cluster(SpectralClustering.knnAffinity(f, 10), 10,
                seed)));
        NeighbourClustering.ScanResult s = NeighbourClustering.scan(features, 10, scan, d.y(), seed);
        out.put("SCAN (clustering step)", score(d.y(), s.scanLabels()));
        out.put("SCAN + self-labelling", score(d.y(), s.labels()));
        out.put("SCAN neighbour purity", new Score(s.neighbourPurity(), Double.NaN, Double.NaN));
        return out;
    }

    private static double[][] unit(float[][] x) {
        double[][] u = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            double s = 1e-12;
            for (float v : x[i]) {
                s += (double) v * v;
            }
            s = Math.sqrt(s);
            u[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                u[i][j] = x[i][j] / s;
            }
        }
        return u;
    }

    private static Tensor images(float[][] x) {
        return Tensor.of(x).reshape(Shape.of(x.length, 1, 8, 8));
    }

    /**
     * The long-budget comparison (a Run): three seeds, 100 images per class, longer training.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (long seed = 1; seed <= 3; seed++) {
            System.out.println("seed " + seed + " " + embedded(100,
                    new DeepEmbeddedClustering.Settings(128, 10, 2000, 2000, 128, 0.1f, 3e-3f, 50), seed));
            System.out.println("seed " + seed + " " + neighbourBased(100, 1000, 2000,
                    new NeighbourClustering.ScanSettings(20, 128, 1000, 300, 0.99, 5, 256), seed));
        }
    }
}
