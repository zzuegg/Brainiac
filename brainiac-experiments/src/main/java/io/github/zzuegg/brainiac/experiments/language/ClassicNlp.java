package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ParallelText;
import io.github.zzuegg.brainiac.experimental.data.SyntheticCorpus;
import io.github.zzuegg.brainiac.experimental.document.LineRecognizer;
import io.github.zzuegg.brainiac.experimental.llm.KneserNeyLm;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.nlp.NoisyChannelCorrector;
import io.github.zzuegg.brainiac.experimental.nlp.SimCse;
import io.github.zzuegg.brainiac.experimental.nlp.WordAlignment;
import io.github.zzuegg.brainiac.experimental.nn.Seq2SeqAttention;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Wave 11 items 11.5.7-11.5.9: classic NLP pieces.
 *
 * <ul>
 *   <li>{@link #alignment} (11.5.7): alignment error rate of IBM 1, IBM 2 and HMM aligners in
 *       both directions and symmetrised, against Dice-coefficient links and the argmax of a
 *       Bahdanau seq2seq model's attention, on {@link ParallelText} pairs with gold links;
 *   <li>{@link #spelling} (11.5.8): word accuracy of noisy-channel correction (uniform,
 *       learned single-edit and Brill-Moore channels, a Kneser-Ney bigram prior) against
 *       uncorrected text, the nearest dictionary word by Levenshtein distance and a character
 *       LM decoded under the channel's logits processor, on a pseudo-word rendering of
 *       {@link SyntheticCorpus} documents with planted systematic typos;
 *   <li>{@link #sentenceEmbeddings} (11.5.9): Spearman correlation with graded role overlap,
 *       paraphrase recall@1 among swapped-role contradictions, alignment and uniformity of
 *       unsupervised and supervised SimCSE against mean static embeddings and a masked-token
 *       encoder.
 * </ul>
 */
public final class ClassicNlp {

    private ClassicNlp() {
    }

    /**
     * A row.
     *
     * @param condition the data set or setting
     * @param method the method
     * @param score AER, accuracy or Spearman
     * @param second a second measure (or {@code NaN})
     */
    public record Row(String condition, String method, double score, double second) {
    }

    // ---------------------------------------------------------------- 11.5.7 alignment

    private static List<int[]> flip(List<int[]> links) {
        List<int[]> out = new ArrayList<>();
        for (int[] l : links) {
            out.add(new int[] {l[1], l[0]});
        }
        return out;
    }

    /**
     * Word alignment.
     *
     * @param sizes training pair counts
     * @param evaluated pairs scored (the first ones; aligners are trained on all, unsupervised)
     * @param iterations EM iterations per model
     * @param seq2seqSteps training steps of the attention baseline (0 skips it), at the first
     *     size only
     * @param seed the seed
     * @return rows: condition = size, score = AER
     */
    public static List<Row> alignment(int[] sizes, int evaluated, int iterations, int seq2seqSteps, long seed) {
        List<Row> rows = new ArrayList<>();
        int sv = ParallelText.sourceVocabulary();
        int tv = ParallelText.targetVocabulary();
        for (int n : sizes) {
            List<ParallelText.Pair> pairs = ParallelText.generate(n, 4, 10, seed);
            List<int[]> src = new ArrayList<>();
            List<int[]> tgt = new ArrayList<>();
            List<List<int[]>> gold = new ArrayList<>();
            for (ParallelText.Pair p : pairs) {
                src.add(p.source());
                tgt.add(p.target());
            }
            int ev = Math.min(evaluated, n);
            for (int i = 0; i < ev; i++) {
                gold.add(pairs.get(i).alignment());
            }
            String cond = n + " pairs";
            for (WordAlignment.Model m : WordAlignment.Model.values()) {
                WordAlignment fwd = WordAlignment.train(m, src, tgt, sv, tv, iterations);
                WordAlignment bwd = WordAlignment.train(m, tgt, src, tv, sv, iterations);
                List<List<int[]>> f = new ArrayList<>();
                List<List<int[]>> b = new ArrayList<>();
                Map<WordAlignment.Symmetrisation, List<List<int[]>>> sym = new HashMap<>();
                for (int i = 0; i < ev; i++) {
                    List<int[]> fl = fwd.align(src.get(i), tgt.get(i));
                    List<int[]> bl = bwd.align(tgt.get(i), src.get(i));
                    f.add(fl);
                    b.add(flip(bl));
                    for (WordAlignment.Symmetrisation s : WordAlignment.Symmetrisation.values()) {
                        sym.computeIfAbsent(s, k -> new ArrayList<>()).add(WordAlignment.symmetrise(fl, bl,
                                src.get(i).length, tgt.get(i).length, s));
                    }
                }
                rows.add(new Row(cond, m + " src->tgt", WordAlignment.aer(f, gold), Double.NaN));
                rows.add(new Row(cond, m + " tgt->src", WordAlignment.aer(b, gold), Double.NaN));
                for (WordAlignment.Symmetrisation s : WordAlignment.Symmetrisation.values()) {
                    rows.add(new Row(cond, m + " " + s, WordAlignment.aer(sym.get(s), gold), Double.NaN));
                }
            }
            List<List<int[]>> dice = WordAlignment.dice(src, tgt, sv, tv, 0.3);
            rows.add(new Row(cond, "Dice", WordAlignment.aer(dice.subList(0, ev), gold), Double.NaN));
            if (seq2seqSteps > 0 && n == sizes[0]) {
                rows.add(new Row(cond, "Bahdanau attention",
                        WordAlignment.aer(attentionLinks(src, tgt, ev, seq2seqSteps, seed), gold), Double.NaN));
            }
        }
        return rows;
    }

    private static List<List<int[]>> attentionLinks(List<int[]> src, List<int[]> tgt, int ev, int steps, long seed) {
        int sv = ParallelText.sourceVocabulary();
        int tv = ParallelText.targetVocabulary();
        Seq2SeqAttention model = Seq2SeqAttention.of(sv, tv, 24, 32, Seq2SeqAttention.Attention.BAHDANAU, seed);
        Map<Integer, List<Integer>> buckets = new HashMap<>();
        for (int i = 0; i < src.size(); i++) {
            buckets.computeIfAbsent(src.get(i).length, k -> new ArrayList<>()).add(i);
        }
        List<Integer> lengths = new ArrayList<>(buckets.keySet());
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        List<Var> params = model.parameters();
        for (int s = 0; s < steps; s++) {
            List<Integer> bucket = buckets.get(lengths.get(r.nextInt(lengths.size())));
            int b = Math.min(16, bucket.size());
            int[][] xs = new int[b][];
            int[][] ys = new int[b][];
            int len = 0;
            for (int i = 0; i < b; i++) {
                int k = bucket.get(r.nextInt(bucket.size()));
                xs[i] = src.get(k);
                ys[i] = tgt.get(k);
                len = Math.max(len, ys[i].length);
            }
            for (int i = 0; i < b; i++) {
                int[] y = Arrays.copyOf(ys[i], len);
                Arrays.fill(y, ys[i].length, len, -1);
                ys[i] = y;
            }
            params.forEach(Var::zeroGrad);
            model.loss(model.embedSource(xs), ys).backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
        List<List<int[]>> out = new ArrayList<>();
        for (int i = 0; i < ev; i++) {
            int[] x = src.get(i);
            Var.noGrad(() -> model.greedy(model.embedSource(new int[][] {x}), tgt.get(out.size()).length));
            float[][] a = model.lastAttention();
            List<int[]> links = new ArrayList<>();
            for (int j = 0; j < a.length; j++) {
                int best = 0;
                for (int k = 1; k < a[j].length; k++) {
                    if (a[j][k] > a[j][best]) {
                        best = k;
                    }
                }
                links.add(new int[] {best, j});
            }
            out.add(links);
        }
        return out;
    }

    // ---------------------------------------------------------------- 11.5.8 spelling

    private static final String CONSONANTS = "bdkmnprst";
    private static final String VOWELS = "aeiou";

    private static List<String> lexicon(int n, SplittableRandom r) {
        Set<String> words = new LinkedHashSet<>();
        while (words.size() < n) {
            StringBuilder w = new StringBuilder();
            int syll = 2 + (r.nextDouble() < 0.2 ? 1 : 0);
            for (int s = 0; s < syll; s++) {
                w.append(CONSONANTS.charAt(r.nextInt(CONSONANTS.length())));
                w.append(VOWELS.charAt(r.nextInt(VOWELS.length())));
            }
            if (r.nextDouble() < 0.4) {
                w.append(CONSONANTS.charAt(r.nextInt(CONSONANTS.length())));
            }
            words.add(w.toString());
        }
        return new ArrayList<>(words);
    }

    /** Systematic typos: confusions dominate, other edits are rare. */
    private static String typo(String w, SplittableRandom r) {
        String[][] confusions = {{"e", "a"}, {"i", "e"}, {"d", "t"}, {"b", "p"}, {"o", "u"}, {"m", "n"}};
        double u = r.nextDouble();
        StringBuilder b = new StringBuilder(w);
        if (u < 0.6) {
            List<int[]> spots = new ArrayList<>();
            for (int i = 0; i < w.length(); i++) {
                for (int c = 0; c < confusions.length; c++) {
                    if (w.substring(i, i + 1).equals(confusions[c][0])) {
                        spots.add(new int[] {i, c});
                    }
                }
            }
            if (!spots.isEmpty()) {
                int[] s = spots.get(r.nextInt(spots.size()));
                b.setCharAt(s[0], confusions[s[1]][1].charAt(0));
                return b.toString();
            }
        }
        if (u < 0.8) {
            int i = r.nextInt(w.length());
            b.insert(i, w.charAt(i));
        } else if (u < 0.9 && w.length() > 2) {
            int i = r.nextInt(w.length() - 1);
            char c = b.charAt(i);
            b.setCharAt(i, b.charAt(i + 1));
            b.setCharAt(i + 1, c);
        } else {
            b.deleteCharAt(r.nextInt(w.length()));
        }
        return b.toString();
    }

    private static WindowLm charLm(List<String> dict, int steps, long seed) {
        WindowLm lm = WindowLm.of(28, 4, 64, seed);
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(5e-3f);
        List<Var> params = lm.parameters();
        for (int s = 0; s < steps; s++) {
            List<int[]> prompts = new ArrayList<>();
            List<int[]> completions = new ArrayList<>();
            for (int i = 0; i < 32; i++) {
                prompts.add(new int[] {0});
                completions.add(NoisyChannelCorrector.charIds(dict.get(r.nextInt(dict.size()))));
            }
            params.forEach(Var::zeroGrad);
            lm.logProbs(prompts, completions).mean().mul(-1f).backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
        return lm;
    }

    /**
     * Spelling correction.
     *
     * @param typoRate the share of misspelt tokens
     * @param charSteps character LM training steps (0 skips that method)
     * @param seed the seed
     * @return rows: condition "typos", score = accuracy on all tokens, second = on misspelt
     *     tokens
     */
    public static List<Row> spelling(double typoRate, int charSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        SyntheticCorpus corpus = SyntheticCorpus.generate(8, 8, 12, 0.5, seed);
        List<String> dict = lexicon(corpus.vocabulary(), r);
        int[][] docs = corpus.documents();
        int split = docs.length * 7 / 10;
        KneserNeyLm lm = KneserNeyLm.train(Arrays.copyOf(docs, split), 2, corpus.vocabulary(), 0.75);
        List<String[]> clean = new ArrayList<>();
        List<String[]> noisy = new ArrayList<>();
        List<boolean[]> wrong = new ArrayList<>();
        for (int d = split; d < docs.length; d++) {
            String[] c = new String[docs[d].length];
            String[] o = new String[c.length];
            boolean[] t = new boolean[c.length];
            for (int k = 0; k < c.length; k++) {
                c[k] = dict.get(docs[d][k]);
                t[k] = r.nextDouble() < typoRate;
                o[k] = t[k] ? typo(c[k], r) : c[k];
            }
            clean.add(c);
            noisy.add(o);
            wrong.add(t);
        }
        // more noisy text for unsupervised channel EM (train documents, never their clean form)
        List<String[]> emText = new ArrayList<>(noisy);
        for (int d = 0; d < split; d++) {
            String[] o = new String[docs[d].length];
            for (int k = 0; k < o.length; k++) {
                String c = dict.get(docs[d][k]);
                o[k] = r.nextDouble() < typoRate ? typo(c, r) : c;
            }
            emText.add(o);
        }
        List<Row> rows = new ArrayList<>();
        rows.add(score("uncorrected", clean, noisy, wrong, (o, k) -> o[k]));
        rows.add(score("nearest Levenshtein", clean, noisy, wrong, (o, k) -> {
            String best = o[k];
            int bd = Integer.MAX_VALUE;
            for (String w : dict) {
                int dd = LineRecognizer.editDistance(w, o[k]);
                if (dd < bd) {
                    bd = dd;
                    best = w;
                }
            }
            return best;
        }));
        NoisyChannelCorrector last = null;
        for (NoisyChannelCorrector.Channel ch : NoisyChannelCorrector.Channel.values()) {
            NoisyChannelCorrector nc = NoisyChannelCorrector.create(dict, lm, ch);
            nc.learnChannel(emText, 3);
            rows.add(score("noisy channel " + ch, clean, noisy, wrong, (o, k) -> nc.correct(o[k],
                    k > 0 ? nc.id(o[k - 1]) : -1, k + 1 < o.length ? nc.id(o[k + 1]) : -1)));
            last = nc;
        }
        if (charSteps > 0) {
            WindowLm clm = charLm(dict, charSteps, seed);
            NoisyChannelCorrector nc = last;
            rows.add(score("char LM + channel processor", clean, noisy, wrong,
                    (o, k) -> nc.decodeWithLm(clm, o[k], 2.0)));
        }
        return rows;
    }

    private interface Fixer {
        String fix(String[] observed, int k);
    }

    private static Row score(String name, List<String[]> clean, List<String[]> noisy, List<boolean[]> wrong,
            Fixer f) {
        int all = 0;
        int ok = 0;
        int typos = 0;
        int fixed = 0;
        for (int s = 0; s < clean.size(); s++) {
            for (int k = 0; k < clean.get(s).length; k++) {
                boolean hit = f.fix(noisy.get(s), k).equals(clean.get(s)[k]);
                all++;
                ok += hit ? 1 : 0;
                if (wrong.get(s)[k] && !noisy.get(s)[k].equals(clean.get(s)[k])) {
                    typos++;
                    fixed += hit ? 1 : 0;
                }
            }
        }
        return new Row("typos", name, ok / (double) all, fixed / (double) Math.max(1, typos));
    }

    // ---------------------------------------------------------------- 11.5.9 SimCSE

    private static final int NOUNS = 12;
    private static final int VERBS = 8;
    private static final int WAS = 2 * NOUNS + 4 * VERBS;
    private static final int BY = WAS + 1;
    private static final int FILLER = BY + 1;
    private static final int SENT_VOCAB = FILLER + 4;

    private static int[] render(int[] t, SplittableRandom r) {
        List<Integer> s = new ArrayList<>();
        int a = 2 * t[0] + r.nextInt(2);
        int o = 2 * t[2] + r.nextInt(2);
        int syn = r.nextInt(2);
        if (r.nextBoolean()) {
            s.add(a);
            s.add(2 * NOUNS + 2 * t[1] + syn);
            s.add(o);
        } else {
            s.add(o);
            s.add(WAS);
            s.add(2 * NOUNS + 2 * VERBS + 2 * t[1] + syn);
            s.add(BY);
            s.add(a);
        }
        if (r.nextBoolean()) {
            s.add(r.nextBoolean() ? 0 : s.size(), FILLER + r.nextInt(4));
        }
        return s.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int[] triple(SplittableRandom r) {
        int a = r.nextInt(NOUNS);
        int o = r.nextInt(NOUNS - 1);
        return new int[] {a, r.nextInt(VERBS), o >= a ? o + 1 : o};
    }

    private static int[] swap(int[] t) {
        return new int[] {t[2], t[1], t[0]};
    }

    private static int[] change(int[] t, int role, SplittableRandom r) {
        int[] u = t.clone();
        int limit = role == 1 ? VERBS : NOUNS;
        do {
            u[role] = r.nextInt(limit);
        } while (u[role] == t[role] || u[0] == u[2]);
        return u;
    }

    private static int overlap(int[] a, int[] b) {
        int s = 0;
        for (int i = 0; i < 3; i++) {
            s += a[i] == b[i] ? 1 : 0;
        }
        return s;
    }

    /**
     * Spearman's rank correlation with average ranks for ties.
     *
     * @param x values
     * @param y values
     * @return the correlation
     */
    public static double spearman(double[] x, double[] y) {
        double[] rx = ranks(x);
        double[] ry = ranks(y);
        double mx = Arrays.stream(rx).average().orElse(0);
        double my = Arrays.stream(ry).average().orElse(0);
        double sxy = 0;
        double sxx = 0;
        double syy = 0;
        for (int i = 0; i < x.length; i++) {
            sxy += (rx[i] - mx) * (ry[i] - my);
            sxx += (rx[i] - mx) * (rx[i] - mx);
            syy += (ry[i] - my) * (ry[i] - my);
        }
        return sxy / Math.sqrt(sxx * syy + 1e-12);
    }

    private static double[] ranks(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(v[a], v[b]));
        double[] r = new double[v.length];
        for (int i = 0; i < v.length;) {
            int j = i;
            while (j + 1 < v.length && v[idx[j + 1]] == v[idx[i]]) {
                j++;
            }
            for (int k = i; k <= j; k++) {
                r[idx[k]] = (i + j) / 2.0;
            }
            i = j + 1;
        }
        return r;
    }

    private interface Embedder {
        float[][] embed(List<int[]> s);
    }

    /**
     * Sentence embeddings.
     *
     * @param trainSentences training sentences (and anchor triples)
     * @param steps training steps per model
     * @param seed the seed
     * @return rows: condition "STS-like", score = Spearman, second = recall@1; plus rows
     *     condition "alignment/uniformity"
     */
    public static List<Row> sentenceEmbeddings(int trainSentences, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> raw = new ArrayList<>();
        List<int[]> anchors = new ArrayList<>();
        List<int[]> positives = new ArrayList<>();
        List<int[]> negatives = new ArrayList<>();
        for (int i = 0; i < trainSentences; i++) {
            int[] t = triple(r);
            raw.add(render(t, r));
            anchors.add(render(t, r));
            positives.add(render(t, r));
            negatives.add(render(r.nextBoolean() ? swap(t) : change(t, r.nextInt(3), r), r));
        }
        // graded pairs
        SplittableRandom er = new SplittableRandom(seed + 7);
        List<int[]> left = new ArrayList<>();
        List<int[]> right = new ArrayList<>();
        List<Double> gold = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            int[] t = triple(er);
            int[] u;
            int kind = er.nextInt(5);
            if (kind == 4) {
                u = swap(t);
            } else {
                u = t.clone();
                List<Integer> roles = new ArrayList<>(List.of(0, 1, 2));
                java.util.Collections.shuffle(roles, new java.util.Random(er.nextLong()));
                for (int k = 0; k < 3 - kind; k++) {
                    u = change(u, roles.get(k), er);
                }
            }
            left.add(render(t, er));
            right.add(render(u, er));
            gold.add((double) overlap(t, u));
        }
        // paraphrase retrieval among contradictions
        List<int[]> queries = new ArrayList<>();
        List<int[]> pool = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        while (queries.size() < 100) {
            int[] t = triple(er);
            if (!seen.add(Arrays.toString(t)) || seen.contains(Arrays.toString(swap(t)))) {
                continue;
            }
            queries.add(render(t, er));
            pool.add(render(t, er));
        }
        int q = queries.size();
        for (int i = 0; i < q; i++) {
            // the swapped twin of query i, rendered in the paraphrase's surface form
            int[] p = pool.get(i);
            pool.add(swapSurface(p));
        }
        List<Row> rows = new ArrayList<>();
        SimCse mlm = new SimCse(SENT_VOCAB, 32, 0.1f, seed);
        mlm.trainMlm(raw, steps, 32, 5e-3f, seed);
        SimCse unsup = new SimCse(SENT_VOCAB, 32, 0.1f, seed);
        unsup.trainUnsupervised(raw, steps, 32, 3e-3f, 0.05f, seed);
        SimCse sup = new SimCse(SENT_VOCAB, 32, 0.1f, seed);
        sup.trainSupervised(anchors, positives, negatives, steps, 32, 3e-3f, 0.05f, seed);
        String[] names = {"mean static embeddings", "MLM encoder", "SimCSE unsupervised", "SimCSE supervised"};
        Embedder[] methods = {mlm::staticEmbed, mlm::embed, unsup::embed, sup::embed};
        double[] g = gold.stream().mapToDouble(Double::doubleValue).toArray();
        for (int m = 0; m < methods.length; m++) {
            float[][] a = methods[m].embed(left);
            float[][] b = methods[m].embed(right);
            double[] cos = new double[a.length];
            for (int i = 0; i < a.length; i++) {
                cos[i] = SimCse.dot(a[i], b[i]);
            }
            float[][] qe = methods[m].embed(queries);
            float[][] pe = methods[m].embed(pool);
            int hit = 0;
            for (int i = 0; i < q; i++) {
                int best = 0;
                for (int k = 1; k < pe.length; k++) {
                    if (SimCse.dot(qe[i], pe[k]) > SimCse.dot(qe[i], pe[best])) {
                        best = k;
                    }
                }
                hit += best == i ? 1 : 0;
            }
            rows.add(new Row("STS-like", names[m], spearman(cos, g), hit / (double) q));
            rows.add(new Row("alignment/uniformity", names[m],
                    SimCse.alignment(qe, Arrays.copyOf(pe, q)), SimCse.uniformity(pe)));
        }
        return rows;
    }

    /** Swaps agent and patient in a rendered sentence, keeping voice, synonyms and filler. */
    private static int[] swapSurface(int[] s) {
        int[] o = s.clone();
        int first = -1;
        for (int k = 0; k < o.length; k++) {
            if (o[k] < 2 * NOUNS) {
                if (first < 0) {
                    first = k;
                } else {
                    int t = o[first];
                    o[first] = o[k];
                    o[k] = t;
                    break;
                }
            }
        }
        return o;
    }
}
