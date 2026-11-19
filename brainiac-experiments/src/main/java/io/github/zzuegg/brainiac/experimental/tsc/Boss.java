package io.github.zzuegg.brainiac.experimental.tsc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Bag-of-SFA-Symbols (BOSS; Schäfer, "The BOSS is concerned with time series classification in the
 * presence of noise", DMKD 2015). Every sliding window becomes an SFA word ({@link SymbolicFourier}
 * with multiple coefficient binning); runs of the same word are counted once (numerosity
 * reduction); a series is its word histogram. Classification is 1-nearest-neighbour under the
 * asymmetric BOSS distance {@code sum over the query's words of (q_w - c_w)^2}, which ignores words
 * absent from the query. The ensemble fits one such model per window length, keeps those whose
 * leave-one-out training accuracy is within {@code keep} of the best, and takes a majority vote.
 * Because only how often patterns occur matters, not where, BOSS is robust to phase shifts and
 * noise where whole-series distances fail.
 */
public final class Boss {

    /**
     * Settings.
     *
     * @param windows the window lengths to try
     * @param wordLengths SFA word lengths tried per window (the best by leave-one-out accuracy is kept)
     * @param alphabet SFA alphabet size
     * @param normMean whether to drop the window mean
     * @param normStd whether to divide windows by their standard deviation (BOSS's default)
     * @param keep members within this fraction of the best leave-one-out accuracy are kept, such as 0.92
     */
    public record Settings(int[] windows, int[] wordLengths, int alphabet, boolean normMean, boolean normStd,
            double keep) {

        /**
         * Defaults for a series length: up to eight window lengths from 10 to half the length, word
         * lengths 4, 6 and 8, alphabet 4, mean and standard deviation normalisation.
         *
         * @param length the series length
         * @return the settings
         */
        public static Settings defaults(int length) {
            int lo = Math.min(10, length / 4);
            int hi = Math.max(lo + 1, length / 2);
            int count = Math.min(8, hi - lo + 1);
            int[] w = new int[count];
            for (int i = 0; i < count; i++) {
                w[i] = lo + (int) Math.round((hi - lo) * i / (double) Math.max(1, count - 1));
            }
            return new Settings(w, new int[] {4, 6, 8}, 4, true, true, 0.92);
        }
    }

    private record Member(SymbolicFourier sfa, List<Map<Integer, Integer>> train, double accuracy) {
    }

    private final List<Member> members;
    private final int[] labels;
    private final int classes;

    private Boss(List<Member> members, int[] labels, int classes) {
        this.members = members;
        this.labels = labels;
        this.classes = classes;
    }

    /**
     * Fits the ensemble.
     *
     * @param series training series
     * @param labels their classes
     * @param classes the number of classes
     * @param settings the settings
     * @return the classifier
     */
    public static Boss fit(float[][] series, int[] labels, int classes, Settings settings) {
        List<Member> all = new ArrayList<>();
        for (int w : settings.windows()) {
            Member best = null;
            for (int wordLength : settings.wordLengths()) {
                SymbolicFourier sfa = SymbolicFourier.fit(series, null, w, wordLength, settings.alphabet(),
                        settings.normMean(), settings.normStd(), SymbolicFourier.Binning.EQUI_DEPTH,
                        Math.max(1, w / 4));
                List<Map<Integer, Integer>> h = new ArrayList<>();
                for (float[] s : series) {
                    h.add(histogram(sfa.words(s)));
                }
                Member m = new Member(sfa, h, leaveOneOut(h, labels));
                if (best == null || m.accuracy() > best.accuracy()) {
                    best = m;
                }
            }
            all.add(best);
        }
        double top = 0;
        for (Member m : all) {
            top = Math.max(top, m.accuracy());
        }
        List<Member> kept = new ArrayList<>();
        for (Member m : all) {
            if (m.accuracy() >= settings.keep() * top) {
                kept.add(m);
            }
        }
        return new Boss(kept, labels.clone(), classes);
    }

    private static double leaveOneOut(List<Map<Integer, Integer>> h, int[] labels) {
        int right = 0;
        for (int i = 0; i < h.size(); i++) {
            double best = Double.POSITIVE_INFINITY;
            int arg = -1;
            for (int j = 0; j < h.size(); j++) {
                if (j != i) {
                    double d = distance(h.get(i), h.get(j), best);
                    if (d < best) {
                        best = d;
                        arg = j;
                    }
                }
            }
            right += arg >= 0 && labels[arg] == labels[i] ? 1 : 0;
        }
        return right / (double) h.size();
    }

    /**
     * Word counts with numerosity reduction (a run of one word counts once).
     *
     * @param words the window words in order
     * @return word to count
     */
    public static Map<Integer, Integer> histogram(int[] words) {
        Map<Integer, Integer> h = new HashMap<>();
        int last = Integer.MIN_VALUE;
        for (int w : words) {
            if (w != last) {
                h.merge(w, 1, Integer::sum);
                last = w;
            }
        }
        return h;
    }

    /**
     * The BOSS distance, stopping early once it exceeds {@code bound}.
     *
     * @param query the query histogram
     * @param candidate the candidate histogram
     * @param bound the best distance so far
     * @return {@code sum_{w in query} (q_w - c_w)^2}, or a value above the bound
     */
    public static double distance(Map<Integer, Integer> query, Map<Integer, Integer> candidate, double bound) {
        double d = 0;
        for (Map.Entry<Integer, Integer> e : query.entrySet()) {
            int c = candidate.getOrDefault(e.getKey(), 0);
            double diff = e.getValue() - c;
            d += diff * diff;
            if (d > bound) {
                return d;
            }
        }
        return d;
    }

    /**
     * Class vote shares per series.
     *
     * @param series the series
     * @return {@code [n][classes]} shares of ensemble votes
     */
    public double[][] probabilities(float[][] series) {
        double[][] out = new double[series.length][classes];
        for (int i = 0; i < series.length; i++) {
            for (Member m : members) {
                Map<Integer, Integer> q = histogram(m.sfa().words(series[i]));
                double best = Double.POSITIVE_INFINITY;
                int arg = 0;
                for (int j = 0; j < m.train().size(); j++) {
                    double d = distance(q, m.train().get(j), best);
                    if (d < best) {
                        best = d;
                        arg = j;
                    }
                }
                out[i][labels[arg]] += 1.0 / members.size();
            }
        }
        return out;
    }

    /**
     * Predicts classes by majority vote.
     *
     * @param series the series
     * @return the classes
     */
    public int[] predict(float[][] series) {
        return WideRidgeClassifier.argmax(probabilities(series));
    }

    /** @return the number of kept window lengths */
    public int size() {
        return members.size();
    }
}
