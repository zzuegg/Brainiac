package io.github.zzuegg.brainiac.experiments.algorithms;

import io.github.zzuegg.brainiac.data.FeatureHasher;
import io.github.zzuegg.brainiac.experimental.data.KeyStreams;
import io.github.zzuegg.brainiac.experimental.data.ReservoirReplay;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.sketch.FrequencySketches;
import io.github.zzuegg.brainiac.experimental.sketch.FrequencySketches.FrequencySketch;
import io.github.zzuegg.brainiac.experimental.sketch.FrequentDirections;
import io.github.zzuegg.brainiac.experimental.sketch.HyperLogLog;
import io.github.zzuegg.brainiac.experimental.sketch.LearnedSketches;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.IntToLongFunction;

/**
 * Streaming sketches (items 11.2.14 and 11.2.15): frequency estimation, heavy hitters and
 * distinct counts on Zipf streams, learning-augmented frequency sketches with a trained
 * heavy-hitter oracle, and Frequent Directions (plain and learned) on matrix streams.
 */
public final class SketchShootout {

    private SketchShootout() {
    }

    /**
     * A frequency estimator's accuracy.
     *
     * @param method the method
     * @param words its memory in counters
     * @param meanError mean absolute error over the distinct items of the stream
     * @param weightedError frequency-weighted absolute error, over the stream length
     * @param precision heavy-hitter precision (items estimated at or above the threshold)
     * @param recall heavy-hitter recall
     */
    public record FreqRow(String method, int words, double meanError, double weightedError, double precision,
            double recall) {
        @Override
        public String toString() {
            return String.format("%s (%d words): mean err %.2f, weighted %.2e, HH P %.2f R %.2f", method, words,
                    meanError, weightedError, precision, recall);
        }
    }

    /**
     * Scores an estimator against the exact counts.
     *
     * @param method its name
     * @param words its memory
     * @param estimate the estimate per item
     * @param counts the exact counts
     * @param heavy how many items are heavy hitters
     * @param length the stream length
     * @return the row
     */
    static FreqRow score(String method, int words, IntToLongFunction estimate, long[] counts, int heavy, long length) {
        long[] sorted = counts.clone();
        Arrays.sort(sorted);
        long threshold = sorted[Math.max(0, sorted.length - heavy)];
        double err = 0;
        double weighted = 0;
        int distinct = 0;
        int tp = 0;
        int reported = 0;
        int actual = 0;
        for (int i = 0; i < counts.length; i++) {
            long e = estimate.applyAsLong(i);
            boolean h = counts[i] >= threshold;
            boolean r = e >= threshold;
            actual += h ? 1 : 0;
            reported += r ? 1 : 0;
            tp += h && r ? 1 : 0;
            if (counts[i] > 0) {
                distinct++;
                err += Math.abs(e - counts[i]);
                weighted += counts[i] * (double) Math.abs(e - counts[i]);
            }
        }
        return new FreqRow(method, words, err / Math.max(1, distinct), weighted / length / length,
                tp / (double) Math.max(1, reported), tp / (double) Math.max(1, actual));
    }

    /**
     * Classic frequency sketches against exact counts, the hashing trick and uniform
     * sampling at equal memory.
     *
     * @param stream the stream
     * @param universe item ids are below this
     * @param words memory in counters
     * @param heavy how many items count as heavy hitters
     * @param seed the seed
     * @return one row per method
     */
    public static List<FreqRow> frequency(int[] stream, int universe, int words, int heavy, long seed) {
        long[] counts = new long[universe];
        for (int x : stream) {
            counts[x]++;
        }
        List<FreqRow> rows = new ArrayList<>();
        int distinct = (int) Arrays.stream(counts).filter(c -> c > 0).count();
        rows.add(score("exact", 2 * distinct, i -> counts[i], counts, heavy, stream.length));
        for (FrequencySketches.Kind k : FrequencySketches.Kind.values()) {
            FrequencySketch s = FrequencySketches.of(k, words, 4, seed);
            for (int x : stream) {
                s.add(x);
            }
            rows.add(score(k.name(), s.words(), s::estimate, counts, heavy, stream.length));
        }
        for (boolean signed : new boolean[] {false, true}) {
            FeatureHasher hasher = new FeatureHasher(words, signed);
            long[] table = new long[words];
            String[] names = new String[universe];
            for (int i = 0; i < universe; i++) {
                names[i] = Integer.toString(i);
            }
            for (int x : stream) {
                table[hasher.column(names[x])] += (long) hasher.sign(names[x]);
            }
            rows.add(score(signed ? "FeatureHasher signed" : "FeatureHasher", words,
                    i -> Math.max(0, table[hasher.column(names[i])] * (long) hasher.sign(names[i])), counts, heavy,
                    stream.length));
        }
        ReservoirReplay reservoir = new ReservoirReplay(words, seed);
        int chunk = 4096;
        for (int from = 0; from < stream.length; from += chunk) {
            int n = Math.min(chunk, stream.length - from);
            float[] x = new float[n];
            for (int i = 0; i < n; i++) {
                x[i] = stream[from + i];
            }
            Tensor t = Tensor.of(Shape.of(n, 1), x);
            reservoir.add(t, Tensor.of(Shape.of(n), x));
        }
        float[] kept = reservoir.toDataset().features().toArray();
        long[] sampled = new long[universe];
        for (float v : kept) {
            sampled[(int) v]++;
        }
        double scale = stream.length / (double) kept.length;
        rows.add(score("reservoir sample", words, i -> Math.round(sampled[i] * scale), counts, heavy, stream.length));
        return rows;
    }

    /**
     * HyperLogLog's relative error at several register counts.
     *
     * @param stream the stream
     * @param ps register bits to try
     * @param seed the hash seed
     * @return per {@code p}: {registers, relative error, theoretical standard error}
     */
    public static Map<Integer, double[]> distinct(int[] stream, int[] ps, long seed) {
        long exact = Arrays.stream(stream).distinct().count();
        Map<Integer, double[]> out = new LinkedHashMap<>();
        for (int p : ps) {
            double err = 0;
            int reps = 5;
            HyperLogLog last = null;
            for (int rep = 0; rep < reps; rep++) {
                HyperLogLog h = new HyperLogLog(p, seed + rep);
                for (int x : stream) {
                    h.add(x);
                }
                err += Math.abs(h.estimate() - exact) / exact;
                last = h;
            }
            out.put(p, new double[] {last.registers(), err / reps, last.standardError()});
        }
        return out;
    }

    /**
     * A Zipf stream whose popularity ranks follow item features: items are ordered by a
     * noisy linear score of {@link SyntheticRecsys} item features, and rank {@code r} of
     * {@link KeyStreams#zipf} is mapped to the {@code r}-th item of that order.
     *
     * @param length the stream length
     * @param universe items
     * @param s the Zipf exponent
     * @param featureNoise noise on the popularity score (relative to its spread)
     * @param seed the seed
     * @return {@code {stream (int[]), features (float[][])}}
     */
    public static Object[] featuredStream(int length, int universe, double s, double featureNoise, long seed) {
        Tensor f = SyntheticRecsys.preferences(1, universe, 8, 16, 0.1, 1, seed).itemFeatures();
        int d = f.shape().dim(1);
        float[] flat = f.toArray();
        float[][] features = new float[universe][d];
        SplittableRandom r = new SplittableRandom(seed + 1);
        double[] w = new double[d];
        for (int j = 0; j < d; j++) {
            w[j] = r.nextGaussian();
        }
        double[] score = new double[universe];
        for (int i = 0; i < universe; i++) {
            System.arraycopy(flat, i * d, features[i], 0, d);
            for (int j = 0; j < d; j++) {
                score[i] += w[j] * features[i][j];
            }
        }
        double sd = Math.sqrt(Arrays.stream(score).map(x -> x * x).average().orElse(1));
        for (int i = 0; i < universe; i++) {
            score[i] += featureNoise * sd * r.nextGaussian();
        }
        Integer[] byScore = new Integer[universe];
        for (int i = 0; i < universe; i++) {
            byScore[i] = i;
        }
        Arrays.sort(byScore, (a, b) -> Double.compare(score[b], score[a]));
        int[] ids = KeyStreams.rankToId(universe, seed);
        int[] rankOf = new int[universe];
        for (int rank = 0; rank < universe; rank++) {
            rankOf[ids[rank]] = rank;
        }
        int[] z = KeyStreams.zipf(length, universe, s, seed);
        int[] out = new int[length];
        for (int i = 0; i < length; i++) {
            out[i] = byScore[rankOf[z[i]]];
        }
        return new Object[] {out, features};
    }

    /**
     * Learned Count-Min and learned Misra-Gries against their classic forms at equal memory,
     * with the trained oracle, an exact one, and noisy versions of the trained one.
     *
     * @param stream the stream
     * @param features per item, its features
     * @param words memory in counters
     * @param heavy predicted-heavy items the oracle is trained to find
     * @param noises flip rates for the trained oracle
     * @param seed the seed
     * @return rows; learned methods are labelled with their oracle
     */
    public static List<FreqRow> learned(int[] stream, float[][] features, int words, int heavy, double[] noises,
            long seed) {
        int universe = features.length;
        long[] counts = new long[universe];
        for (int x : stream) {
            counts[x]++;
        }
        int[] prefix = Arrays.copyOf(stream, stream.length / 10);
        LearnedSketches.HeavyOracle trained = LearnedSketches.classifier(features, prefix, heavy, 300, seed);
        long[] sorted = counts.clone();
        Arrays.sort(sorted);
        long cut = sorted[universe - heavy];
        LearnedSketches.HeavyOracle exact = i -> counts[i] >= cut;
        Map<String, LearnedSketches.HeavyOracle> oracles = new LinkedHashMap<>();
        oracles.put("exact oracle", exact);
        oracles.put("trained", trained);
        for (double n : noises) {
            oracles.put("trained, " + (int) (100 * n) + "% flipped", LearnedSketches.noisy(trained, n, seed + 3));
        }
        List<FreqRow> rows = new ArrayList<>();
        for (FrequencySketches.Kind k : new FrequencySketches.Kind[] {FrequencySketches.Kind.COUNT_MIN,
                FrequencySketches.Kind.CONSERVATIVE_UPDATE, FrequencySketches.Kind.MISRA_GRIES}) {
            FrequencySketch s = FrequencySketches.of(k, words, 4, seed);
            for (int x : stream) {
                s.add(x);
            }
            rows.add(score(k.name(), s.words(), s::estimate, counts, heavy, stream.length));
        }
        for (Map.Entry<String, LearnedSketches.HeavyOracle> o : oracles.entrySet()) {
            for (LearnedSketches.Kind k : LearnedSketches.Kind.values()) {
                FrequencySketch s = LearnedSketches.of(k, words, 4, o.getValue(), 0.3, seed);
                for (int x : stream) {
                    s.add(x);
                }
                rows.add(score(k.name() + " [" + o.getKey() + "]", s.words(), s::estimate, counts, heavy,
                        stream.length));
            }
        }
        return rows;
    }

    /**
     * Matrix sketches of a stream of {@link SyntheticRecsys} user rows: plain and learned
     * Frequent Directions against the offline SVD, row sampling and Oja's rule. The learned
     * sketch's directions are the top singular vectors of the first tenth of the stream (or
     * random ones).
     *
     * @param users rows in the stream
     * @param ell sketch rows
     * @param predicted reserved predicted directions
     * @param drift whether the second half of the stream comes from a different population
     * @param seed the seed
     * @return method to relative covariance error {@code ||A^T A - B^T B||_2 / ||A||_F^2}
     */
    public static Map<String, Double> matrix(int users, int ell, int predicted, boolean drift, long seed) {
        double[][] a = userRows(users, drift, seed);
        int d = a[0].length;
        double[][] gram = FrequentDirections.gram(a);
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("SVD oracle rank " + ell, FrequentDirections.covarianceError(gram, FrequentDirections.svdOracle(a, ell)));
        FrequentDirections fd = new FrequentDirections(ell, d);
        for (double[] row : a) {
            fd.add(row);
        }
        out.put("FD", FrequentDirections.covarianceError(gram, fd.covariance()));
        double[][] past = Arrays.copyOf(a, Math.max(predicted + 1, users / 10));
        double[][] dirs = FrequentDirections.topDirections(past, predicted);
        FrequentDirections.Learned lfd = FrequentDirections.learned(ell, dirs);
        for (double[] row : a) {
            lfd.add(row);
        }
        out.put("learned FD", FrequentDirections.covarianceError(gram, lfd.covariance()));
        SplittableRandom r = new SplittableRandom(seed + 9);
        double[][] noise = new double[users / 10][d];
        for (double[] row : noise) {
            for (int j = 0; j < d; j++) {
                row[j] = r.nextGaussian();
            }
        }
        FrequentDirections.Learned rfd = FrequentDirections.learned(ell, FrequentDirections.topDirections(noise, predicted));
        for (double[] row : a) {
            rfd.add(row);
        }
        out.put("learned FD, random directions", FrequentDirections.covarianceError(gram, rfd.covariance()));
        out.put("row sampling", FrequentDirections.covarianceError(gram, FrequentDirections.rowSampling(a, ell, seed)));
        out.put("SVD oracle rank 1", FrequentDirections.covarianceError(gram, FrequentDirections.svdOracle(a, 1)));
        out.put("Oja rank 1", FrequentDirections.covarianceError(gram, FrequentDirections.oja(a, 0.01, seed)));
        return out;
    }

    private static double[][] userRows(int users, boolean drift, long seed) {
        int half = drift ? users / 2 : users;
        float[] first = SyntheticRecsys.preferences(half, 10, 4, 32, 0.3, 1, seed).userFeatures().toArray();
        float[] second = drift ? SyntheticRecsys.preferences(users - half, 10, 4, 32, 0.3, 1, seed + 101)
                .userFeatures().toArray() : new float[0];
        int d = 32;
        double[][] a = new double[users][d];
        for (int i = 0; i < users; i++) {
            for (int j = 0; j < d; j++) {
                a[i][j] = i < half ? first[i * d + j] : second[(i - half) * d + j];
            }
        }
        return a;
    }
}
