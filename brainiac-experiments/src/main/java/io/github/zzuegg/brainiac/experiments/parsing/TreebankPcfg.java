package io.github.zzuegg.brainiac.experiments.parsing;

import io.github.zzuegg.brainiac.experimental.data.ToyTreebank;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A treebank PCFG: rules and their maximum-likelihood probabilities read off the training trees
 * (right-factored into binary rules with intermediate symbols, keeping the unary
 * {@code NP -> Name} and {@code VP -> Vi} rules), word emissions per tag with add-0.1 smoothing,
 * and Viterbi CKY. It has the generator's own grammar family but no lexical preferences above the
 * tags.
 */
final class TreebankPcfg {

    private final Map<String, Integer> symbols = new HashMap<>();
    private final List<String> names = new ArrayList<>();
    private final List<int[]> binary = new ArrayList<>();      // {parent, left, right}
    private final List<Double> binaryLogp = new ArrayList<>();
    private final List<int[]> unary = new ArrayList<>();       // {parent, child}
    private final List<Double> unaryLogp = new ArrayList<>();
    private final double[][] emit;                              // [tag][word]

    TreebankPcfg(List<ToyTreebank.Sentence> train) {
        for (String t : ToyTreebank.TAGS) {
            id(t);
        }
        Map<String, Double> ruleCounts = new HashMap<>();
        Map<Integer, Double> lhsCounts = new HashMap<>();
        int tags = ToyTreebank.TAGS.size();
        int vocab = ToyTreebank.VOCABULARY.size();
        emit = new double[tags][vocab];
        for (ToyTreebank.Sentence s : train) {
            count(s.tree(), ruleCounts, lhsCounts);
            for (int i = 0; i < s.length(); i++) {
                emit[s.tags()[i]][s.words()[i]] += 1;
            }
        }
        for (double[] row : emit) {
            double total = 0;
            for (double v : row) {
                total += v + 0.1;
            }
            for (int w = 0; w < row.length; w++) {
                row[w] = Math.log((row[w] + 0.1) / total);
            }
        }
        ruleCounts.forEach((k, v) -> {
            String[] p = k.split(" ");
            int lhs = id(p[0]);
            double logp = Math.log(v / lhsCounts.get(lhs));
            if (p.length == 2) {
                unary.add(new int[] {lhs, id(p[1])});
                unaryLogp.add(logp);
            } else {
                binary.add(new int[] {lhs, id(p[1]), id(p[2])});
                binaryLogp.add(logp);
            }
        });
    }

    private int id(String s) {
        return symbols.computeIfAbsent(s, k -> {
            names.add(k);
            return names.size() - 1;
        });
    }

    private void count(ToyTreebank.Tree t, Map<String, Double> rules, Map<Integer, Double> lhs) {
        if (t.preterminal()) {
            return;
        }
        List<String> kids = t.children().stream().map(ToyTreebank.Tree::label).toList();
        String parent = t.label();
        if (kids.size() == 1) {
            add(parent, List.of(kids.get(0)), rules, lhs);
        }
        for (int i = 0; i + 1 < kids.size(); i++) {
            String right = i + 2 == kids.size() ? kids.get(i + 1)
                    : t.label() + "|" + String.join("_", kids.subList(i + 1, kids.size()));
            add(parent, List.of(kids.get(i), right), rules, lhs);
            parent = right;
        }
        for (ToyTreebank.Tree k : t.children()) {
            count(k, rules, lhs);
        }
    }

    private void add(String parent, List<String> rhs, Map<String, Double> rules, Map<Integer, Double> lhs) {
        rules.merge(parent + " " + String.join(" ", rhs), 1.0, Double::sum);
        lhs.merge(id(parent), 1.0, Double::sum);
    }

    /**
     * Viterbi parse.
     *
     * @param words word ids
     * @return labelled constituents {@code {phrase, start, end}}, or {@code null} if no parse
     */
    List<int[]> parse(int[] words) {
        int n = words.length;
        int m = names.size();
        double[][][] best = new double[n + 1][n + 1][m];
        int[][][] back = new int[n + 1][n + 1][m];      // split * m*m + left * m + right, or -1 - child
        for (double[][] a : best) {
            for (double[] b : a) {
                java.util.Arrays.fill(b, Double.NEGATIVE_INFINITY);
            }
        }
        int tags = ToyTreebank.TAGS.size();
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < tags; t++) {
                best[i][i + 1][t] = emit[t][words[i]];
            }
            for (int u = 0; u < unary.size(); u++) {
                int[] r = unary.get(u);
                double v = unaryLogp.get(u) + best[i][i + 1][r[1]];
                if (v > best[i][i + 1][r[0]]) {
                    best[i][i + 1][r[0]] = v;
                    back[i][i + 1][r[0]] = -1 - r[1];
                }
            }
        }
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i + len <= n; i++) {
                int j = i + len;
                double[] cell = best[i][j];
                for (int k = i + 1; k < j; k++) {
                    double[] left = best[i][k];
                    double[] right = best[k][j];
                    for (int q = 0; q < binary.size(); q++) {
                        int[] r = binary.get(q);
                        double v = left[r[1]] + right[r[2]];
                        if (v == Double.NEGATIVE_INFINITY) {
                            continue;
                        }
                        v += binaryLogp.get(q);
                        if (v > cell[r[0]]) {
                            cell[r[0]] = v;
                            back[i][j][r[0]] = (k * m + r[1]) * m + r[2];
                        }
                    }
                }
            }
        }
        int s = symbols.get("S");
        if (best[0][n][s] == Double.NEGATIVE_INFINITY) {
            return null;
        }
        Map<Long, int[]> spans = new java.util.LinkedHashMap<>();
        collect(0, n, s, back, spans);
        return new ArrayList<>(spans.values());
    }

    private void collect(int i, int j, int sym, int[][][] back, Map<Long, int[]> spans) {
        int m = names.size();
        String name = names.get(sym);
        int tags = ToyTreebank.TAGS.size();
        if (sym < tags) {
            return;
        }
        int code = back[i][j][sym];
        if (code < 0) {
            collect(i, j, -1 - code, back, spans);
        } else {
            int k = code / (m * m);
            collect(i, k, code / m % m, back, spans);
            collect(k, j, code % m, back, spans);
        }
        int phrase = ToyTreebank.PHRASES.indexOf(name);
        if (phrase >= 0) {
            spans.put((long) i * 1000 + j, new int[] {phrase, i, j});
        }
    }
}
