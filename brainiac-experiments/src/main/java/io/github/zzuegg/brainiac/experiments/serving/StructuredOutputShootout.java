package io.github.zzuegg.brainiac.experiments.serving;

import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.experimental.serving.GrammarMaskCache;
import io.github.zzuegg.brainiac.llm.Constraint;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Serving shootout, structured-output rows (8.4.9): grammar-constrained greedy decoding of a
 * toy LM (seeded random logits, with a bias toward ending once the output is complete) under
 * two GBNF grammars of the core {@code Grammar} (arithmetic with nested brackets, and JSON
 * objects with nested arrays), with a byte-level vocabulary and a BPE vocabulary (the core
 * {@link BytePairTokenizer}, 256 merges learned on sample outputs). Mask methods:
 * <ul>
 * <li>per-token scan: every token's text fed to the matcher from scratch;</li>
 * <li>the core {@link Constraint.Vocabulary#allowed} prefix-sharing scan (what the existing
 *     constrained decoders call each step);</li>
 * <li>{@link GrammarMaskCache}, lazily filled, and filled ahead of time
 *     ({@link GrammarMaskCache#precompute}).</li>
 * </ul>
 * All must give identical outputs. Reported: mask time per generated token, cache hit rate and
 * cached states.
 */
public final class StructuredOutputShootout {

    /** Arithmetic with nested brackets. */
    public static final String ARITHMETIC = """
            root ::= expr
            expr ::= term (("+" | "-" | "*") term)*
            term ::= num | "(" expr ")"
            num ::= [0-9] [0-9]?
            """;

    /** JSON objects with string, number and nested array values. */
    public static final String JSON = """
            root ::= "{" pair ("," pair)* "}"
            pair ::= "\\"" [a-e]+ "\\":" value
            value ::= [0-9]+ | "\\"" [a-e]* "\\"" | "[" (value ("," value)*)? "]"
            """;

    private StructuredOutputShootout() {
    }

    /**
     * One mask method on one grammar and vocabulary.
     *
     * @param microsPerToken mask time per generated token
     * @param hitRate cache lookups answered without a scan (0 for scans)
     * @param states automaton states cached
     * @param identical whether the outputs equal the per-token scan's
     */
    public record Row(double microsPerToken, double hitRate, int states, boolean identical) {
    }

    static String sample(String grammar, SplittableRandom r, int depth) {
        if (grammar.equals(ARITHMETIC)) {
            return expr(r, depth);
        }
        StringBuilder sb = new StringBuilder("{");
        int pairs = 1 + r.nextInt(3);
        for (int i = 0; i < pairs; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append('"').append(word(r, 1)).append("\":").append(value(r, depth));
        }
        return sb.append('}').toString();
    }

    private static String expr(SplittableRandom r, int depth) {
        StringBuilder sb = new StringBuilder(term(r, depth));
        int n = r.nextInt(3);
        for (int i = 0; i < n; i++) {
            sb.append("+-*".charAt(r.nextInt(3))).append(term(r, depth));
        }
        return sb.toString();
    }

    private static String term(SplittableRandom r, int depth) {
        return depth > 0 && r.nextInt(3) == 0 ? "(" + expr(r, depth - 1) + ")" : Integer.toString(r.nextInt(100));
    }

    private static String word(SplittableRandom r, int min) {
        StringBuilder sb = new StringBuilder();
        int n = min + r.nextInt(4);
        for (int i = 0; i < n; i++) {
            sb.append((char) ('a' + r.nextInt(5)));
        }
        return sb.toString();
    }

    private static String value(SplittableRandom r, int depth) {
        int k = r.nextInt(depth > 0 ? 3 : 2);
        if (k == 0) {
            return Integer.toString(r.nextInt(1000));
        }
        if (k == 1) {
            return "\"" + word(r, 0) + "\"";
        }
        StringBuilder sb = new StringBuilder("[");
        int n = r.nextInt(3);
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(value(r, depth - 1));
        }
        return sb.append(']').toString();
    }

    /**
     * The token texts of a vocabulary.
     *
     * @param grammar the grammar the BPE corpus is drawn from
     * @param bpe whether to learn BPE merges (otherwise ASCII bytes)
     * @param seed the seed
     * @return the token texts
     */
    static String[] vocabulary(String grammar, boolean bpe, long seed) {
        if (!bpe) {
            String[] texts = new String[128];
            for (int c = 32; c < 127; c++) {
                texts[c] = String.valueOf((char) c);
            }
            return texts;
        }
        SplittableRandom r = new SplittableRandom(seed);
        StringBuilder corpus = new StringBuilder();
        for (int i = 0; i < 2000; i++) {
            corpus.append(sample(grammar, r, 2));
        }
        BytePairTokenizer tok = BytePairTokenizer.train(corpus.toString(), 512);
        String[] texts = new String[tok.size()];
        for (int t = 0; t < texts.length; t++) {
            byte[] b = tok.tokenBytes(t);
            boolean ascii = true;
            for (byte x : b) {
                ascii &= x >= 32 && x < 127;
            }
            texts[t] = ascii ? new String(b, java.nio.charset.StandardCharsets.US_ASCII) : null;
        }
        return texts;
    }

    private interface Masker {
        boolean[] mask(Constraint.Matcher m);
    }

    /**
     * Runs the structured-output rows.
     *
     * @param outputs outputs per grammar and vocabulary
     * @param maxTokens the most tokens per output
     * @param seed the seed
     * @return "grammar / vocabulary" to method to row
     */
    public static Map<String, Map<String, Row>> run(int outputs, int maxTokens, long seed) {
        Map<String, Map<String, Row>> out = new LinkedHashMap<>();
        for (String g : List.of(ARITHMETIC, JSON)) {
            Constraint c = Constraint.grammar(g);
            for (boolean bpe : new boolean[] {false, true}) {
                String[] texts = vocabulary(g, bpe, seed);
                Constraint.Vocabulary vocab = new Constraint.Vocabulary(texts);
                Map<String, Masker> methods = new LinkedHashMap<>();
                methods.put("per-token scan", m -> {
                    boolean[] ok = new boolean[texts.length];
                    for (int t = 0; t < texts.length; t++) {
                        ok[t] = texts[t] != null && !texts[t].isEmpty() && m.feed(texts[t]) != null;
                    }
                    return ok;
                });
                methods.put("core Vocabulary prefix scan", vocab::allowed);
                GrammarMaskCache lazy = GrammarMaskCache.of(vocab);
                methods.put("GrammarMaskCache (lazy)", lazy::allowed);
                GrammarMaskCache ahead = GrammarMaskCache.of(vocab);
                ahead.precompute(c.start(), 400);
                methods.put("GrammarMaskCache (precomputed)", ahead::allowed);
                Map<String, Row> rows = new LinkedHashMap<>();
                List<int[]> reference = null;
                for (int pass = 0; pass < 2; pass++) {   // the first pass warms up the JIT
                    for (Map.Entry<String, Masker> e : methods.entrySet()) {
                        long[] nanos = new long[1];
                        long[] tokens = new long[1];
                        List<int[]> gen = generate(c, vocab, e.getValue(), outputs, maxTokens, seed + pass, nanos,
                                tokens);
                        if (reference == null || pass == 1 && e.getKey().equals("per-token scan")) {
                            reference = gen;
                        }
                        boolean same = gen.size() == reference.size();
                        for (int i = 0; same && i < gen.size(); i++) {
                            same = Arrays.equals(gen.get(i), reference.get(i));
                        }
                        GrammarMaskCache cache = e.getKey().contains("lazy") ? lazy
                                : e.getKey().contains("precomputed") ? ahead : null;
                        double hit = cache == null ? 0 : cache.hits() / (double) Math.max(1, cache.hits()
                                + cache.misses());
                        rows.put(e.getKey(), new Row(nanos[0] / 1e3 / tokens[0], hit,
                                cache == null ? 0 : cache.states(), same));
                    }
                }
                out.put((g.equals(ARITHMETIC) ? "arithmetic" : "JSON") + " / " + (bpe ? "BPE " + texts.length
                        : "bytes"), rows);
            }
        }
        return out;
    }

    private static List<int[]> generate(Constraint c, Constraint.Vocabulary vocab, Masker masker, int outputs,
            int maxTokens, long seed, long[] nanos, long[] tokens) {
        List<int[]> out = new ArrayList<>();
        int v = vocab.size();
        for (int o = 0; o < outputs; o++) {
            SplittableRandom r = new SplittableRandom(seed * 1000 + o);
            Constraint.Matcher m = c.start();
            List<Integer> seq = new ArrayList<>();
            for (int step = 0; step < maxTokens; step++) {
                long t0 = System.nanoTime();
                boolean[] ok = masker.mask(m);
                nanos[0] += System.nanoTime() - t0;
                tokens[0]++;
                int best = -1;
                double bestScore = m.accepting() ? 1.5 + r.nextGaussian() * 0.5 : Double.NEGATIVE_INFINITY;
                for (int t = 0; t < v; t++) {
                    double s = r.nextGaussian();
                    if (ok[t] && s > bestScore) {
                        bestScore = s;
                        best = t;
                    }
                }
                if (best < 0) {
                    break;   // end of sequence chosen (or nothing else allowed)
                }
                seq.add(best);
                m = vocab.advance(m, best);
            }
            out.add(seq.stream().mapToInt(Integer::intValue).toArray());
        }
        return out;
    }
}
