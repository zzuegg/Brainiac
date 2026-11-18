package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Synthetic data streams with concept drift at known positions, the standard benchmarks of
 * stream mining (MOA):
 * <ul>
 *   <li>{@link #sea SEA}: three features in {@code [0, 10]}, label {@code x0 + x1 <= theta}, the
 *       threshold switching between 8, 9, 7 and 9.5 (Street and Kim, 2001);</li>
 *   <li>{@link #hyperplane rotating hyperplane}: label {@code w . x >= sum(w) / 2} for uniform
 *       {@code x}, a new weight vector per concept (Hulten et al., 2001);</li>
 *   <li>{@link #rbf RBF}: Gaussian clusters with class labels; each concept moves the centroids
 *       and reassigns their classes;</li>
 *   <li>{@link #clicks drifting clicks}: the click model of {@link SyntheticRecsys#clicks}
 *       (a hidden interaction between two categorical fields), with a new interaction matrix per
 *       concept, delivered as hashed sparse features (one per field plus every field pair).</li>
 * </ul>
 * A {@link Drift} schedule switches concepts at given positions: {@code ABRUPT} switches at once,
 * {@code GRADUAL} draws each sample from the new concept with probability
 * {@code sigmoid(4 (t - p) / width)} around each position {@code p} (MOA's concept-drift stream),
 * and {@code RECURRING} alternates between the first two concepts.
 */
public final class DriftingStreams {

    /** How concepts change at the drift positions. */
    public enum Drift {
        /** Switch at once. */
        ABRUPT,
        /** Mix old and new over a window around each position. */
        GRADUAL,
        /** Alternate between two concepts (abruptly). */
        RECURRING
    }

    /**
     * One labelled example.
     *
     * @param x dense features, or {@code null} for sparse-only streams
     * @param active indices of the active binary features, or {@code null} for dense streams
     * @param label the class
     * @param concept the concept that generated it
     */
    public record Sample(double[] x, int[] active, int label, int concept) {
    }

    /** A stream of samples. */
    public interface Stream {

        /** @return the next sample */
        Sample next();

        /** @return dense features per sample, or the hashed feature space for sparse streams */
        int features();

        /** @return the number of classes */
        int classes();

        /** @return the drift positions */
        int[] driftPoints();
    }

    /** Produces a sample of one concept. */
    @FunctionalInterface
    private interface Concept {
        Sample sample(int concept, SplittableRandom r);
    }

    private static final class Scheduled implements Stream {
        private final Concept concept;
        private final Drift drift;
        private final int[] points;
        private final int width;
        private final int features;
        private final int classes;
        private final SplittableRandom random;
        private int t;

        Scheduled(Concept concept, Drift drift, int[] points, int width, int features, int classes, long seed) {
            this.concept = concept;
            this.drift = drift;
            this.points = points.clone();
            this.width = Math.max(1, width);
            this.features = features;
            this.classes = classes;
            this.random = new SplittableRandom(seed);
        }

        @Override
        public Sample next() {
            int c = 0;
            for (int i = 0; i < points.length; i++) {
                boolean switched = drift == Drift.GRADUAL
                        ? random.nextDouble() < 1 / (1 + Math.exp(-4.0 * (t - points[i]) / width))
                        : t >= points[i];
                if (!switched) {
                    break;
                }
                c = drift == Drift.RECURRING ? (i + 1) % 2 : i + 1;
            }
            t++;
            return concept.sample(c, random);
        }

        @Override
        public int features() {
            return features;
        }

        @Override
        public int classes() {
            return classes;
        }

        @Override
        public int[] driftPoints() {
            return points.clone();
        }
    }

    private DriftingStreams() {
    }

    /**
     * The SEA concepts stream.
     *
     * @param drift the drift kind
     * @param points drift positions
     * @param width the gradual-drift width
     * @param noise the label-flip probability, such as 0.1
     * @param seed the seed
     * @return the stream
     */
    public static Stream sea(Drift drift, int[] points, int width, double noise, long seed) {
        double[] thetas = {8, 9, 7, 9.5};
        return new Scheduled((c, r) -> {
            double[] x = {10 * r.nextDouble(), 10 * r.nextDouble(), 10 * r.nextDouble()};
            int y = x[0] + x[1] <= thetas[c % thetas.length] ? 1 : 0;
            if (r.nextDouble() < noise) {
                y = 1 - y;
            }
            return new Sample(x, null, y, c);
        }, drift, points, width, 3, 2, seed);
    }

    /**
     * The hyperplane stream.
     *
     * @param dims features
     * @param drift the drift kind
     * @param points drift positions
     * @param width the gradual-drift width
     * @param noise the label-flip probability
     * @param seed the seed
     * @return the stream
     */
    public static Stream hyperplane(int dims, Drift drift, int[] points, int width, double noise, long seed) {
        SplittableRandom setup = new SplittableRandom(seed ^ 0x5DEECE66DL);
        double[][] w = new double[points.length + 1][dims];
        for (double[] row : w) {
            for (int i = 0; i < dims; i++) {
                row[i] = setup.nextDouble();
            }
        }
        return new Scheduled((c, r) -> {
            double[] x = new double[dims];
            double s = 0;
            double half = 0;
            for (int i = 0; i < dims; i++) {
                x[i] = r.nextDouble();
                s += w[c][i] * x[i];
                half += w[c][i] / 2;
            }
            int y = s >= half ? 1 : 0;
            if (r.nextDouble() < noise) {
                y = 1 - y;
            }
            return new Sample(x, null, y, c);
        }, drift, points, width, dims, 2, seed);
    }

    /**
     * The RBF stream: {@code centroids} Gaussian clusters, each with a class; every concept
     * draws new centroid positions and classes.
     *
     * @param dims features
     * @param centroids clusters
     * @param classes classes
     * @param drift the drift kind
     * @param points drift positions
     * @param width the gradual-drift width
     * @param seed the seed
     * @return the stream
     */
    public static Stream rbf(int dims, int centroids, int classes, Drift drift, int[] points, int width, long seed) {
        SplittableRandom setup = new SplittableRandom(seed ^ 0x9E3779B97F4A7C15L);
        int concepts = points.length + 1;
        double[][][] centre = new double[concepts][centroids][dims];
        int[][] label = new int[concepts][centroids];
        double[] std = new double[centroids];
        for (int k = 0; k < centroids; k++) {
            std[k] = 0.05 + 0.1 * setup.nextDouble();
        }
        for (int c = 0; c < concepts; c++) {
            for (int k = 0; k < centroids; k++) {
                for (int i = 0; i < dims; i++) {
                    centre[c][k][i] = setup.nextDouble();
                }
                label[c][k] = k % classes;
            }
            // shuffle the classes of the clusters
            for (int k = centroids - 1; k > 0; k--) {
                int j = setup.nextInt(k + 1);
                int tmp = label[c][k];
                label[c][k] = label[c][j];
                label[c][j] = tmp;
            }
        }
        return new Scheduled((c, r) -> {
            int k = r.nextInt(centroids);
            double[] x = new double[dims];
            for (int i = 0; i < dims; i++) {
                x[i] = centre[c][k][i] + std[k] * r.nextGaussian();
            }
            return new Sample(x, null, label[c][k], c);
        }, drift, points, width, dims, classes, seed);
    }

    /**
     * The drifting click stream: each impression has {@code fields} categorical fields; the
     * click probability is {@code sigmoid(2 P_c[a0][a1] + b_c[a_last] - 1)} with a fresh
     * interaction matrix {@code P_c} and bias {@code b_c} per concept. Features are hashed into
     * {@code 2^hashBits} slots: one per field value and one per pair of field values.
     *
     * @param fields categorical fields, at least 2
     * @param categories categories per field
     * @param hashBits the hashed feature space is {@code 2^hashBits}
     * @param drift the drift kind
     * @param points drift positions
     * @param width the gradual-drift width
     * @param seed the seed
     * @return the stream
     */
    public static Stream clicks(int fields, int categories, int hashBits, Drift drift, int[] points, int width,
            long seed) {
        SplittableRandom setup = new SplittableRandom(seed ^ 0xC2B2AE3D27D4EB4FL);
        int concepts = points.length + 1;
        double[][][] pair = new double[concepts][categories][categories];
        double[][] bias = new double[concepts][categories];
        for (int c = 0; c < concepts; c++) {
            for (int a = 0; a < categories; a++) {
                bias[c][a] = 0.5 * setup.nextGaussian();
                for (int b = 0; b < categories; b++) {
                    pair[c][a][b] = setup.nextGaussian();
                }
            }
        }
        int mask = (1 << hashBits) - 1;
        int pairs = fields * (fields - 1) / 2;
        return new Scheduled((c, r) -> {
            int[] cat = new int[fields];
            for (int f = 0; f < fields; f++) {
                cat[f] = r.nextInt(categories);
            }
            int[] active = new int[fields + pairs];
            int k = 0;
            for (int f = 0; f < fields; f++) {
                active[k++] = hash(f, cat[f]) & mask;
                for (int g = f + 1; g < fields; g++) {
                    active[k++] = hash(1000 + f * fields + g, cat[f] * categories + cat[g]) & mask;
                }
            }
            double logit = 2 * pair[c][cat[0]][cat[1]] + bias[c][cat[fields - 1]] - 1;
            int y = r.nextDouble() < 1 / (1 + Math.exp(-logit)) ? 1 : 0;
            return new Sample(null, active, y, c);
        }, drift, points, width, 1 << hashBits, 2, seed);
    }

    /**
     * Hashes a (namespace, value) pair to an integer (the murmur3 finaliser over both), for the
     * hashing trick.
     *
     * @param namespace the feature's field or cross
     * @param value its value
     * @return a well-mixed hash
     */
    public static int hash(int namespace, int value) {
        long h = namespace * 0x9E3779B97F4A7C15L + value;
        h ^= h >>> 33;
        h *= 0xff51afd7ed558ccdL;
        h ^= h >>> 33;
        h *= 0xc4ceb9fe1a85ec53L;
        h ^= h >>> 33;
        return (int) h & Integer.MAX_VALUE;
    }

    /**
     * Draws the first {@code n} samples of a stream.
     *
     * @param stream the stream
     * @param n samples
     * @return them, in order
     */
    public static Sample[] take(Stream stream, int n) {
        Sample[] out = new Sample[n];
        for (int i = 0; i < n; i++) {
            out[i] = stream.next();
        }
        return out;
    }

    /**
     * The concept index of every sample, for checking where drift really happened.
     *
     * @param samples samples
     * @return their concepts
     */
    public static int[] concepts(Sample[] samples) {
        return Arrays.stream(samples).mapToInt(Sample::concept).toArray();
    }
}
