package io.github.zzuegg.brainiac.experimental.nlp;

import io.github.zzuegg.brainiac.experimental.llm.KneserNeyLm;
import io.github.zzuegg.brainiac.experimental.llm.LogitMath;
import io.github.zzuegg.brainiac.experimental.llm.LogitsProcessor;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Noisy-channel spelling correction: the correction of an observed word {@code w} is the
 * dictionary word {@code c} maximising {@code P(c | context) P(w | c)}. Candidates are the
 * dictionary words within Damerau-Levenshtein distance 2 (and {@code w} itself if it is a
 * word); the prior is a word-level {@link KneserNeyLm} over the previous word, times the
 * probability of the next word given {@code c}; the channel is one of
 *
 * <ul>
 *   <li>{@link Channel#UNIFORM}: every edit equally likely ({@code p^edits});
 *   <li>{@link Channel#SINGLE_EDIT}: Kernighan, Church and Gale's confusion probabilities of
 *       single-character substitutions, insertions, deletions and transpositions, each
 *       normalised by the frequency of its source characters;
 *   <li>{@link Channel#BRILL_MOORE}: Brill and Moore's substring rewrites {@code alpha -> beta}
 *       (up to three characters, with one character of context and the position at the
 *       start, middle or end of the word), so realistic errors such as "ph" to "f" or "ent"
 *       to "ant" are single events.
 * </ul>
 *
 * <p>Both learned channels are estimated by expectation maximisation from unlabelled text
 * ({@link #learnChannel}): each observed token's posterior over its candidates weights the
 * edits that turn each candidate into the token, and the edit counts are renormalised.
 *
 * <p>{@link #decodeWithLm} instead spells the correction character by character from a
 * character-level {@link WindowLm} through its {@link LogitsProcessor} hook: the processor
 * keeps only characters that extend a dictionary prefix and adds the channel's log-penalty
 * for the edit distance between the prefix and the closest prefix of the observed word.
 *
 * <p>Papers: Kernighan, Church, Gale, "A Spelling Correction Program Based on a Noisy Channel
 * Model", COLING 1990, <a href="https://aclanthology.org/C90-2036/">C90-2036</a>; Brill, Moore,
 * "An Improved Error Model for Noisy Channel Spelling Correction", ACL 2000,
 * <a href="https://aclanthology.org/P00-1037/">P00-1037</a>.
 */
public final class NoisyChannelCorrector {

    /** Channel models. */
    public enum Channel {
        /** All edits equally likely. */
        UNIFORM,
        /** Single-character confusion probabilities. */
        SINGLE_EDIT,
        /** Substring rewrites with context and position. */
        BRILL_MOORE
    }

    private final List<String> dictionary;
    private final Map<String, Integer> ids = new HashMap<>();
    private final KneserNeyLm lm;
    private final Channel channel;
    private final Map<String, Double> rewrite = new HashMap<>();
    private final Map<String, Double> sourceCounts = new HashMap<>();
    private double editPrior = 0.01;
    private final Set<String> prefixes = new HashSet<>();

    private NoisyChannelCorrector(List<String> dictionary, KneserNeyLm lm, Channel channel) {
        this.dictionary = List.copyOf(dictionary);
        this.lm = lm;
        this.channel = channel;
        for (int i = 0; i < dictionary.size(); i++) {
            ids.put(dictionary.get(i), i);
            String w = dictionary.get(i);
            for (int k = 0; k <= w.length(); k++) {
                prefixes.add(w.substring(0, k));
            }
        }
    }

    /**
     * Creates a corrector.
     *
     * @param dictionary the words; word id {@code i} of the language model is
     *     {@code dictionary.get(i)}
     * @param lm the word-level prior
     * @param channel the channel model
     * @return the corrector (learned channels start uniform)
     */
    public static NoisyChannelCorrector create(List<String> dictionary, KneserNeyLm lm, Channel channel) {
        return new NoisyChannelCorrector(dictionary, lm, channel);
    }

    /**
     * @param word a string
     * @return its dictionary id, or {@code -1}
     */
    public int id(String word) {
        return ids.getOrDefault(word, -1);
    }

    // ------------------------------------------------------------------ edit alignment

    /**
     * The edit operations of an optimal Damerau-Levenshtein alignment from {@code c} to
     * {@code w}, as {@code {from, to, position}} strings, matches included as identity
     * operations.
     *
     * @param c the intended word
     * @param w the observed word
     * @return the operations in order
     */
    public static List<String[]> alignment(String c, String w) {
        int n = c.length();
        int m = w.length();
        int[][] d = new int[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            d[i][0] = i;
        }
        for (int j = 0; j <= m; j++) {
            d[0][j] = j;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                int cost = c.charAt(i - 1) == w.charAt(j - 1) ? 0 : 1;
                d[i][j] = Math.min(d[i - 1][j - 1] + cost, Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1));
                if (i > 1 && j > 1 && c.charAt(i - 1) == w.charAt(j - 2) && c.charAt(i - 2) == w.charAt(j - 1)) {
                    d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
                }
            }
        }
        List<String[]> ops = new ArrayList<>();
        int i = n;
        int j = m;
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + (c.charAt(i - 1) == w.charAt(j - 1) ? 0 : 1)) {
                ops.add(0, new String[] {c.substring(i - 1, i), w.substring(j - 1, j)});
                i--;
                j--;
            } else if (i > 1 && j > 1 && c.charAt(i - 1) == w.charAt(j - 2) && c.charAt(i - 2) == w.charAt(j - 1)
                    && d[i][j] == d[i - 2][j - 2] + 1) {
                ops.add(0, new String[] {c.substring(i - 2, i), w.substring(j - 2, j)});
                i -= 2;
                j -= 2;
            } else if (i > 0 && d[i][j] == d[i - 1][j] + 1) {
                ops.add(0, new String[] {c.substring(i - 1, i), ""});
                i--;
            } else {
                ops.add(0, new String[] {"", w.substring(j - 1, j)});
                j--;
            }
        }
        return ops;
    }

    /**
     * Damerau-Levenshtein distance (adjacent transpositions count once).
     *
     * @param a one string
     * @param b another
     * @return the distance
     */
    public static int distance(String a, String b) {
        int edits = 0;
        for (String[] op : alignment(a, b)) {
            edits += op[0].equals(op[1]) ? 0 : 1;
        }
        return edits;
    }

    /** The rewrite events of an alignment for the chosen channel, with their source strings. */
    private List<String[]> events(String c, String w) {
        List<String[]> ops = alignment(c, w);
        List<String[]> out = new ArrayList<>();
        if (channel == Channel.BRILL_MOORE) {
            int k = 0;
            int pos = 0;
            while (k < ops.size()) {
                if (ops.get(k)[0].equals(ops.get(k)[1])) {
                    pos += ops.get(k)[0].length();
                    k++;
                    continue;
                }
                StringBuilder from = new StringBuilder();
                StringBuilder to = new StringBuilder();
                int start = pos;
                while (k < ops.size() && !ops.get(k)[0].equals(ops.get(k)[1]) && from.length() < 3) {
                    from.append(ops.get(k)[0]);
                    to.append(ops.get(k)[1]);
                    pos += ops.get(k)[0].length();
                    k++;
                }
                String left = start > 0 ? c.substring(start - 1, start) : "^";
                String where = start == 0 ? "B" : pos >= c.length() ? "E" : "M";
                out.add(new String[] {where + left + from, where + left + to, where + left + from});
            }
            return out;
        }
        for (String[] op : ops) {
            if (!op[0].equals(op[1])) {
                out.add(new String[] {op[0], op[1], op[0].isEmpty() ? "" : op[0]});
            }
        }
        return out;
    }

    /**
     * The channel probability {@code log P(w | c)}.
     *
     * @param c the intended word
     * @param w the observed word
     * @return the log-probability
     */
    public double logChannel(String c, String w) {
        if (c.equals(w)) {
            return Math.log(0.95);
        }
        double lp = Math.log(0.05);
        for (String[] e : events(c, w)) {
            double p;
            if (channel == Channel.UNIFORM || rewrite.isEmpty()) {
                p = editPrior;
            } else {
                double num = rewrite.getOrDefault(e[0] + "\u0000" + e[1], 0.0);
                double den = sourceCounts.getOrDefault(e[2], 0.0);
                p = (num + editPrior) / (den + 1);
            }
            lp += Math.log(p);
        }
        return lp;
    }

    /**
     * The candidates of a word: dictionary words within distance 2.
     *
     * @param w the observed word
     * @return their ids
     */
    public List<Integer> candidates(String w) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < dictionary.size(); i++) {
            String c = dictionary.get(i);
            if (Math.abs(c.length() - w.length()) <= 2 && distance(c, w) <= 2) {
                out.add(i);
            }
        }
        return out;
    }

    private double logPrior(int c, int prev, int next) {
        double p = Math.log(lm.probability(prev >= 0 ? new int[] {prev} : new int[0], c) + 1e-12);
        if (next >= 0) {
            p += Math.log(lm.probability(new int[] {c}, next) + 1e-12);
        }
        return p;
    }

    /**
     * Corrects one word in context.
     *
     * @param w the observed word
     * @param prev the previous word's id, or {@code -1}
     * @param next the next word's id, or {@code -1}
     * @return the correction (the word itself without candidates)
     */
    public String correct(String w, int prev, int next) {
        List<Integer> cands = candidates(w);
        String best = w;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int c : cands) {
            double s = logPrior(c, prev, next) + logChannel(dictionary.get(c), w);
            if (s > bestScore) {
                bestScore = s;
                best = dictionary.get(c);
            }
        }
        return best;
    }

    /**
     * Learns the channel by EM from unlabelled sentences of observed (possibly misspelt)
     * words: each token's posterior over its candidates weights the rewrite events that turn
     * the candidate into it.
     *
     * @param sentences observed sentences
     * @param iterations EM iterations
     */
    public void learnChannel(List<String[]> sentences, int iterations) {
        if (channel == Channel.UNIFORM) {
            return;
        }
        Map<String, List<Integer>> cache = new HashMap<>();
        for (int it = 0; it < iterations; it++) {
            Map<String, Double> rw = new HashMap<>();
            Map<String, Double> src = new HashMap<>();
            for (String[] s : sentences) {
                for (int k = 0; k < s.length; k++) {
                    String w = s[k];
                    List<Integer> cands = cache.computeIfAbsent(w, this::candidates);
                    if (cands.isEmpty()) {
                        continue;
                    }
                    int prev = k > 0 ? id(s[k - 1]) : -1;
                    int next = k + 1 < s.length ? id(s[k + 1]) : -1;
                    double[] score = new double[cands.size()];
                    double max = Double.NEGATIVE_INFINITY;
                    for (int q = 0; q < score.length; q++) {
                        score[q] = logPrior(cands.get(q), prev, next) + logChannel(dictionary.get(cands.get(q)), w);
                        max = Math.max(max, score[q]);
                    }
                    double z = 0;
                    for (int q = 0; q < score.length; q++) {
                        score[q] = Math.exp(score[q] - max);
                        z += score[q];
                    }
                    for (int q = 0; q < score.length; q++) {
                        double post = score[q] / z;
                        String c = dictionary.get(cands.get(q));
                        countSources(c, post, src);
                        for (String[] e : events(c, w)) {
                            rw.merge(e[0] + "\u0000" + e[1], post, Double::sum);
                        }
                    }
                }
            }
            rewrite.clear();
            rewrite.putAll(rw);
            sourceCounts.clear();
            sourceCounts.putAll(src);
        }
    }

    /** Counts the source strings an event can rewrite: characters and, for Brill-Moore, contexts. */
    private void countSources(String c, double weight, Map<String, Double> src) {
        if (channel == Channel.BRILL_MOORE) {
            for (int start = 0; start < c.length(); start++) {
                String left = start > 0 ? c.substring(start - 1, start) : "^";
                for (int len = 0; len <= 3 && start + len <= c.length(); len++) {
                    String where = start == 0 ? "B" : start + len >= c.length() ? "E" : "M";
                    src.merge(where + left + c.substring(start, start + len), weight, Double::sum);
                }
            }
            return;
        }
        src.merge("", weight * (c.length() + 1), Double::sum);
        for (int i = 0; i < c.length(); i++) {
            src.merge(c.substring(i, i + 1), weight, Double::sum);
            if (i + 1 < c.length()) {
                src.merge(c.substring(i, i + 2), weight, Double::sum);
            }
        }
    }

    // ------------------------------------------------------------------ character LM decoding

    /** The character ids: 0 begin, 1 end, letters from 2. */
    public static int charId(char ch) {
        return ch - 'a' + 2;
    }

    /**
     * Turns a word into a character LM completion (letters then the end token).
     *
     * @param w a lower-case word
     * @return the ids
     */
    public static int[] charIds(String w) {
        int[] out = new int[w.length() + 1];
        for (int i = 0; i < w.length(); i++) {
            out[i] = charId(w.charAt(i));
        }
        out[w.length()] = 1;
        return out;
    }

    private static int prefixDistance(String p, String w) {
        int best = Integer.MAX_VALUE;
        for (int k = 0; k <= w.length(); k++) {
            best = Math.min(best, distance(p, w.substring(0, k)));
        }
        return best;
    }

    /**
     * A processor that keeps the character LM on dictionary prefixes within channel distance
     * 2 of the observed word and adds {@code -penalty} per edit.
     *
     * @param w the observed word
     * @param penalty the log-penalty per edit
     * @return the processor
     */
    public LogitsProcessor channelProcessor(String w, double penalty) {
        return (context, logits) -> {
            StringBuilder p = new StringBuilder();
            for (int k = 1; k < context.length; k++) {
                p.append((char) ('a' + context[k] - 2));
            }
            String prefix = p.toString();
            for (int tok = 0; tok < logits.length; tok++) {
                if (tok == 0) {
                    logits[tok] = LogitMath.MASKED;
                    continue;
                }
                int d;
                if (tok == 1) {
                    d = ids.containsKey(prefix) ? distance(prefix, w) : 99;
                } else {
                    String q = prefix + (char) ('a' + tok - 2);
                    d = prefixes.contains(q) ? prefixDistance(q, w) : 99;
                }
                logits[tok] = d > 2 ? LogitMath.MASKED : (float) (logits[tok] - penalty * d);
            }
        };
    }

    /**
     * Corrects a word by greedy character-level decoding from a character LM constrained by
     * {@link #channelProcessor}.
     *
     * @param charLm a character LM over {@code 28} ids (begin, end, a-z), trained on
     *     dictionary words
     * @param w the observed word
     * @param penalty the log-penalty per edit
     * @return the decoded word
     */
    public String decodeWithLm(WindowLm charLm, String w, double penalty) {
        int[] out = charLm.sampleUntil(new int[][] {{0}}, w.length() + 3, new int[] {1}, 0, new SplittableRandom(0),
                channelProcessor(w, penalty))[0];
        StringBuilder s = new StringBuilder();
        for (int tok : out) {
            if (tok >= 2) {
                s.append((char) ('a' + tok - 2));
            }
        }
        return s.toString();
    }
}
