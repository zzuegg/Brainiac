package io.github.zzuegg.brainiac.experimental.explain;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Certifiably optimal rule lists, CORELS (Angelino, Larus-Stone, Alabi, Seltzer and Rudin, JMLR
 * 2018, <a href="https://arxiv.org/abs/1704.01701">arXiv:1704.01701</a>): an ordered list of
 * "if antecedent then class" rules and a default class, minimising the training error plus
 * {@code lambda} per rule, found by branch and bound over prefixes.
 *
 * <p>Antecedents are pre-mined literals and conjunctions of two literals (on different binary
 * features) whose support is at least {@code minSupport}. Prefixes are explored best-first by
 * their lower bound (the mistakes the prefix already makes plus {@code lambda} per rule) with the
 * paper's bounds: a prefix whose bound plus {@code lambda} cannot beat the best list is not
 * extended; a new rule must capture at least {@code lambda n} rows not captured before and
 * classify at least {@code lambda n} of them correctly (the minimum and accurate support
 * bounds); and a map from captured row sets to the best bound seen (the prefix permutation map,
 * keyed by the captured set, so it also merges equivalent-support prefixes) drops prefixes that
 * capture the same rows no better. When the queue empties the list is certified optimal.
 */
public final class RuleLists {

    /**
     * An antecedent: one literal, or a conjunction of two.
     *
     * @param feature the first feature
     * @param value its required value
     * @param feature2 the second feature, or -1
     * @param value2 its required value
     */
    public record Antecedent(int feature, boolean value, int feature2, boolean value2) {

        /**
         * @param x a binary row
         * @return whether the antecedent holds
         */
        public boolean holds(double[] x) {
            return (x[feature] > 0.5) == value && (feature2 < 0 || (x[feature2] > 0.5) == value2);
        }

        @Override
        public String toString() {
            String a = (value ? "" : "!") + "b" + feature;
            return feature2 < 0 ? a : a + "&" + (value2 ? "" : "!") + "b" + feature2;
        }
    }

    /**
     * A rule list.
     *
     * @param rules the antecedents in order
     * @param predictions each rule's class
     * @param defaultPrediction the class when no rule fires
     * @param objective training error plus {@code lambda} per rule
     * @param certified whether the search finished (the list is optimal)
     * @param nodes prefixes evaluated
     * @param millis search time
     */
    public record RuleList(List<Antecedent> rules, int[] predictions, int defaultPrediction, double objective,
            boolean certified, long nodes, double millis) {

        /**
         * @param x a binary row
         * @return the class
         */
        public int predict(double[] x) {
            for (int k = 0; k < rules.size(); k++) {
                if (rules.get(k).holds(x)) {
                    return predictions[k];
                }
            }
            return defaultPrediction;
        }

        @Override
        public String toString() {
            StringBuilder b = new StringBuilder();
            for (int k = 0; k < rules.size(); k++) {
                b.append(k == 0 ? "if " : " elif ").append(rules.get(k)).append(" -> ").append(predictions[k]);
            }
            return b.append(rules.isEmpty() ? "" : " else ").append(defaultPrediction).toString();
        }
    }

    private record Node(int[] prefix, int[] predictions, long[] captured, int mistakes, double bound) {
    }

    private record CapturedKey(long[] bits) {
        @Override
        public boolean equals(Object o) {
            return o instanceof CapturedKey k && Arrays.equals(k.bits, bits);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(bits);
        }
    }

    private RuleLists() {
    }

    /**
     * Mines the antecedents.
     *
     * @param x binary rows
     * @param minSupport the least share of rows an antecedent must cover
     * @param pairs whether to include conjunctions of two literals
     * @return the antecedents
     */
    public static List<Antecedent> mine(double[][] x, double minSupport, boolean pairs) {
        int p = x[0].length;
        List<Antecedent> out = new ArrayList<>();
        for (int f = 0; f < p; f++) {
            for (boolean v : new boolean[] {true, false}) {
                Antecedent a = new Antecedent(f, v, -1, false);
                if (support(a, x) >= minSupport) {
                    out.add(a);
                }
            }
        }
        if (pairs) {
            for (int f = 0; f < p; f++) {
                for (int g = f + 1; g < p; g++) {
                    for (boolean v : new boolean[] {true, false}) {
                        for (boolean w : new boolean[] {true, false}) {
                            Antecedent a = new Antecedent(f, v, g, w);
                            if (support(a, x) >= minSupport) {
                                out.add(a);
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    private static double support(Antecedent a, double[][] x) {
        int c = 0;
        for (double[] row : x) {
            if (a.holds(row)) {
                c++;
            }
        }
        return c / (double) x.length;
    }

    /**
     * Searches for the optimal rule list.
     *
     * @param x binary rows
     * @param y 0/1 labels
     * @param antecedents the candidate antecedents
     * @param lambda the penalty per rule
     * @param maxLength the most rules
     * @param maxNodes the search budget
     * @return the best list found (certified if the search finished)
     */
    public static RuleList fit(double[][] x, double[] y, List<Antecedent> antecedents, double lambda, int maxLength,
            long maxNodes) {
        long t0 = System.nanoTime();
        int n = x.length;
        int words = (n + 63) / 64;
        long[][] bits = new long[antecedents.size()][words];
        long[] pos = new long[words];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < bits.length; a++) {
                if (antecedents.get(a).holds(x[i])) {
                    bits[a][i >> 6] |= 1L << (i & 63);
                }
            }
            if (y[i] > 0.5) {
                pos[i >> 6] |= 1L << (i & 63);
            }
        }
        long[] full = new long[words];
        for (int i = 0; i < n; i++) {
            full[i >> 6] |= 1L << (i & 63);
        }
        int totalPos = count(pos);
        double minCount = lambda * n;
        int[] bestPrefix = new int[0];
        int[] bestPred = new int[0];
        int bestDefault = 2 * totalPos > n ? 1 : 0;
        double best = Math.min(totalPos, n - totalPos) / (double) n;
        PriorityQueue<Node> queue = new PriorityQueue<>((a, b) -> Double.compare(a.bound, b.bound));
        queue.add(new Node(new int[0], new int[0], new long[words], 0, 0));
        Map<CapturedKey, Double> permutationMap = new HashMap<>();
        long nodes = 0;
        boolean certified = true;
        long[] fresh = new long[words];
        while (!queue.isEmpty()) {
            if (nodes >= maxNodes) {
                certified = false;
                break;
            }
            Node node = queue.poll();
            if (node.bound + lambda >= best) {
                continue;
            }
            for (int a = 0; a < bits.length; a++) {
                if (contains(node.prefix, a)) {
                    continue;
                }
                nodes++;
                int captured = 0;
                int capturedPos = 0;
                for (int w = 0; w < words; w++) {
                    fresh[w] = bits[a][w] & ~node.captured[w];
                    captured += Long.bitCount(fresh[w]);
                    capturedPos += Long.bitCount(fresh[w] & pos[w]);
                }
                if (captured < minCount) {
                    continue;
                }
                int prediction = 2 * capturedPos > captured ? 1 : 0;
                int correct = prediction == 1 ? capturedPos : captured - capturedPos;
                if (correct < minCount) {
                    continue;
                }
                int mistakes = node.mistakes + captured - correct;
                int length = node.prefix.length + 1;
                double bound = mistakes / (double) n + lambda * length;
                if (bound >= best) {
                    continue;
                }
                long[] cap = new long[words];
                int restPos = 0;
                int rest = 0;
                for (int w = 0; w < words; w++) {
                    cap[w] = node.captured[w] | bits[a][w];
                    long left = full[w] & ~cap[w];
                    rest += Long.bitCount(left);
                    restPos += Long.bitCount(left & pos[w]);
                }
                int defaultPrediction = 2 * restPos > rest ? 1 : 0;
                double objective = bound + Math.min(restPos, rest - restPos) / (double) n;
                int[] prefix = Arrays.copyOf(node.prefix, length);
                prefix[length - 1] = a;
                int[] preds = Arrays.copyOf(node.predictions, length);
                preds[length - 1] = prediction;
                if (objective < best - 1e-12) {
                    best = objective;
                    bestPrefix = prefix;
                    bestPred = preds;
                    bestDefault = defaultPrediction;
                }
                CapturedKey key = new CapturedKey(cap);
                Double seen = permutationMap.get(key);
                if (seen != null && seen <= bound) {
                    continue;
                }
                permutationMap.put(key, bound);
                if (length < maxLength) {
                    queue.add(new Node(prefix, preds, cap, mistakes, bound));
                }
            }
        }
        List<Antecedent> rules = new ArrayList<>();
        for (int a : bestPrefix) {
            rules.add(antecedents.get(a));
        }
        return new RuleList(rules, bestPred, bestDefault, best, certified, nodes, (System.nanoTime() - t0) / 1e6);
    }

    private static boolean contains(int[] prefix, int a) {
        for (int v : prefix) {
            if (v == a) {
                return true;
            }
        }
        return false;
    }

    private static int count(long[] s) {
        int c = 0;
        for (long w : s) {
            c += Long.bitCount(w);
        }
        return c;
    }
}
