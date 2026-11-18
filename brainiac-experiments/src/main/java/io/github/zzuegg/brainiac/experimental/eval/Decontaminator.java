package io.github.zzuegg.brainiac.experimental.eval;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

/**
 * Detectors that flag training documents overlapping a benchmark, after Yang et al. ("Rethinking
 * Benchmark and Contamination for Language Models with Rephrased Samples", 2023,
 * <a href="https://arxiv.org/abs/2311.04850">arXiv:2311.04850</a>). Paraphrased test items pass
 * n-gram decontamination yet still inflate scores; Yang et al. use an LLM to judge candidate
 * pairs, here an embedding-similarity detector stands in.
 * <ul>
 *   <li>{@link #ngram}: flag a document sharing any token n-gram with a benchmark item (the
 *       usual 13-gram rule, at a toy n);</li>
 *   <li>{@link #embedding}: flag a document whose embedding has cosine similarity above a
 *       threshold with its nearest benchmark item.</li>
 * </ul>
 */
public final class Decontaminator {

    private Decontaminator() {
    }

    private static Set<Long> grams(int[] doc, int n) {
        Set<Long> s = new HashSet<>();
        for (int i = 0; i + n <= doc.length; i++) {
            long h = 1469598103934665603L;
            for (int k = 0; k < n; k++) {
                h = (h ^ doc[i + k]) * 1099511628211L;
            }
            s.add(h);
        }
        return s;
    }

    /**
     * N-gram overlap.
     *
     * @param training documents to check
     * @param benchmark the benchmark items
     * @param n the n-gram length
     * @return a flag per training document
     */
    public static boolean[] ngram(List<int[]> training, List<int[]> benchmark, int n) {
        Set<Long> bench = new HashSet<>();
        for (int[] b : benchmark) {
            bench.addAll(grams(b, n));
        }
        boolean[] flag = new boolean[training.size()];
        for (int i = 0; i < flag.length; i++) {
            for (long g : grams(training.get(i), n)) {
                if (bench.contains(g)) {
                    flag[i] = true;
                    break;
                }
            }
        }
        return flag;
    }

    /**
     * The largest cosine similarity of each training document's embedding with any benchmark
     * item's embedding.
     *
     * @param training documents to check
     * @param benchmark the benchmark items
     * @param embed the embedding
     * @return the nearest-neighbour similarity per training document
     */
    public static double[] nearestSimilarity(List<int[]> training, List<int[]> benchmark,
            Function<int[], double[]> embed) {
        double[][] b = new double[benchmark.size()][];
        for (int i = 0; i < b.length; i++) {
            b[i] = unit(embed.apply(benchmark.get(i)));
        }
        double[] out = new double[training.size()];
        for (int i = 0; i < out.length; i++) {
            double[] t = unit(embed.apply(training.get(i)));
            double best = -1;
            for (double[] v : b) {
                double s = 0;
                for (int k = 0; k < v.length; k++) {
                    s += v[k] * t[k];
                }
                best = Math.max(best, s);
            }
            out[i] = best;
        }
        return out;
    }

    /**
     * Embedding nearest-neighbour detection.
     *
     * @param training documents to check
     * @param benchmark the benchmark items
     * @param embed the embedding
     * @param threshold the cosine similarity above which a document is flagged
     * @return a flag per training document
     */
    public static boolean[] embedding(List<int[]> training, List<int[]> benchmark, Function<int[], double[]> embed,
            double threshold) {
        double[] s = nearestSimilarity(training, benchmark, embed);
        boolean[] flag = new boolean[s.length];
        for (int i = 0; i < s.length; i++) {
            flag[i] = s[i] > threshold;
        }
        return flag;
    }

    private static double[] unit(double[] v) {
        double n = 0;
        for (double x : v) {
            n += x * x;
        }
        n = Math.sqrt(Math.max(1e-24, n));
        double[] u = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            u[i] = v[i] / n;
        }
        return u;
    }
}
