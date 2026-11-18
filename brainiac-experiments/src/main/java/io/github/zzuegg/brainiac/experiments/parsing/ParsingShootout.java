package io.github.zzuegg.brainiac.experiments.parsing;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ToyTreebank;
import io.github.zzuegg.brainiac.experimental.nn.LinearChainCrf;
import io.github.zzuegg.brainiac.experimental.nn.PointerNetwork;
import io.github.zzuegg.brainiac.experimental.parse.BiaffineParser;
import io.github.zzuegg.brainiac.experimental.parse.ChartParser;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 10 items 10.5.16 and 10.5.17 on {@link ToyTreebank}: graph-based dependency parsing with a
 * biaffine scorer and tree decoders, and span-based constituency parsing with CKY, against simple
 * baselines. Test sentences may be longer than any training sentence.
 * <ul>
 *   <li>{@link #dependency}: biaffine + Chu-Liu-Edmonds / Eisner / greedy against a pointer
 *       network pointing to each word's head, a BiLSTM-CRF tagging relative head offsets, and
 *       right-branching chains. Metrics: UAS, LAS, the share of well-formed trees, accuracy on
 *       ambiguous PP attachments, and UAS on test sentences longer than the training ones.</li>
 *   <li>{@link #constituency}: the chart parser (BiLSTM with margin or tree-CRF loss, factored
 *       self-attention with the margin loss) against a treebank PCFG with CKY and right-branching
 *       trees. Metrics: labelled bracketing F1, exact match, well-formed share, PP attachment (the
 *       smallest constituent containing the PP) and F1 on the longer sentences.</li>
 * </ul>
 * Seq2seq baselines generating linearised trees were not run.
 */
public final class ParsingShootout {

    /**
     * Sizes.
     *
     * @param train training sentences
     * @param test test sentences
     * @param maxTrain longest training sentence
     * @param maxTest longest test sentence
     * @param steps optimiser steps per model
     * @param batch sentences per step
     */
    public record Budget(int train, int test, int maxTrain, int maxTest, int steps, int batch) {
    }

    private ParsingShootout() {
    }

    private static int[][] words(List<ToyTreebank.Sentence> s) {
        return s.stream().map(ToyTreebank.Sentence::words).toArray(int[][]::new);
    }

    /**
     * The dependency comparison.
     *
     * @param budget sizes
     * @param seed the seed
     * @return per model {@code {UAS, LAS, well-formed, PP accuracy, UAS on long, ms}}
     */
    public static Map<String, double[]> dependency(Budget budget, long seed) {
        List<ToyTreebank.Sentence> train = ToyTreebank.sample(budget.train(), 4, budget.maxTrain(), seed);
        List<ToyTreebank.Sentence> test = ToyTreebank.sample(budget.test(), 4, budget.maxTest(), seed + 1);
        int vocab = ToyTreebank.VOCABULARY.size();
        int labels = ToyTreebank.RELATIONS.size();
        Map<String, double[]> out = new LinkedHashMap<>();
        long t0 = System.currentTimeMillis();
        BiaffineParser parser = new BiaffineParser(vocab, labels, 32, 48, 48, 24, seed);
        parser.train(words(train), train.stream().map(ToyTreebank.Sentence::heads).toArray(int[][]::new),
                train.stream().map(ToyTreebank.Sentence::relations).toArray(int[][]::new), budget.steps(),
                budget.batch(), 3e-3f, seed);
        long ms = System.currentTimeMillis() - t0;
        for (BiaffineParser.Decoder d : BiaffineParser.Decoder.values()) {
            out.put("biaffine " + d.name().toLowerCase(java.util.Locale.ROOT),
                    scoreDependencies(test, budget.maxTrain(), s -> parser.parse(s.words(), d), ms));
        }
        out.put("right-branching", scoreDependencies(test, budget.maxTrain(), s -> {
            int n = s.length();
            int[] h = new int[n];
            for (int i = 0; i < n; i++) {
                h[i] = i + 1 < n ? i + 2 : 0;
            }
            return new int[][] {h, null};
        }, 0));
        t0 = System.currentTimeMillis();
        Function<ToyTreebank.Sentence, int[][]> pointer = pointerBaseline(train, budget, seed);
        out.put("pointer network", scoreDependencies(test, budget.maxTrain(), pointer,
                System.currentTimeMillis() - t0));
        t0 = System.currentTimeMillis();
        Function<ToyTreebank.Sentence, int[][]> crf = offsetCrfBaseline(train, budget, seed);
        out.put("BiLSTM-CRF offsets", scoreDependencies(test, budget.maxTrain(), crf, System.currentTimeMillis() - t0));
        return out;
    }

    private static double[] scoreDependencies(List<ToyTreebank.Sentence> test, int maxTrain,
            Function<ToyTreebank.Sentence, int[][]> parse, long ms) {
        double arcs = 0;
        double uas = 0;
        double las = 0;
        double trees = 0;
        double pp = 0;
        double ppTotal = 0;
        double longArcs = 0;
        double longUas = 0;
        boolean labelled = true;
        for (ToyTreebank.Sentence s : test) {
            int[][] p = parse.apply(s);
            labelled &= p[1] != null;
            trees += BiaffineParser.isTree(p[0]) ? 1 : 0;
            for (int i = 0; i < s.length(); i++) {
                boolean ok = p[0][i] == s.heads()[i];
                arcs++;
                uas += ok ? 1 : 0;
                las += ok && p[1] != null && p[1][i] == s.relations()[i] ? 1 : 0;
                if (s.length() > maxTrain) {
                    longArcs++;
                    longUas += ok ? 1 : 0;
                }
            }
            for (int c : s.ppCases()) {
                ppTotal++;
                pp += p[0][c] == s.heads()[c] ? 1 : 0;
            }
        }
        return new double[] {uas / arcs, labelled ? las / arcs : Double.NaN, trees / test.size(), pp / ppTotal,
            longArcs == 0 ? Double.NaN : longUas / longArcs, ms};
    }

    private static int[] sameLength(List<ToyTreebank.Sentence> train, int batch, SplittableRandom r) {
        int first = r.nextInt(train.size());
        int n = train.get(first).length();
        List<Integer> same = new ArrayList<>();
        for (int i = 0; i < train.size(); i++) {
            if (train.get(i).length() == n) {
                same.add(i);
            }
        }
        int[] out = new int[batch];
        out[0] = first;
        for (int i = 1; i < batch; i++) {
            out[i] = same.get(r.nextInt(same.size()));
        }
        return out;
    }

    /** One-hot words with ROOT (id {@code vocab}) at position 0: {@code [batch, n + 1, vocab + 1]}. */
    private static Var oneHotWithRoot(List<ToyTreebank.Sentence> batch) {
        int v = ToyTreebank.VOCABULARY.size() + 1;
        int n = batch.get(0).length() + 1;
        float[] x = new float[batch.size() * n * v];
        for (int b = 0; b < batch.size(); b++) {
            x[(b * n) * v + v - 1] = 1;
            for (int i = 1; i < n; i++) {
                x[(b * n + i) * v + batch.get(b).words()[i - 1]] = 1;
            }
        }
        return Var.of(Tensor.of(Shape.of(batch.size(), n, v), x));
    }

    private static Function<ToyTreebank.Sentence, int[][]> pointerBaseline(List<ToyTreebank.Sentence> train,
            Budget budget, long seed) {
        PointerNetwork net = PointerNetwork.of(ToyTreebank.VOCABULARY.size() + 1, 48, false, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 7);
        for (int s = 0; s < budget.steps(); s++) {
            List<ToyTreebank.Sentence> batch = new ArrayList<>();
            for (int i : sameLength(train, budget.batch(), r)) {
                batch.add(train.get(i));
            }
            int[][] targets = batch.stream().map(ToyTreebank.Sentence::heads).toArray(int[][]::new);
            Var loss = net.loss(oneHotWithRoot(batch), targets);
            loss.backward();
            opt.step(net.parameters());
        }
        return s -> new int[][] {Var.noGrad(() -> net.decode(oneHotWithRoot(List.of(s)), s.length(), false))[0], null};
    }

    private static final int OFFSET = 12;

    private static Function<ToyTreebank.Sentence, int[][]> offsetCrfBaseline(List<ToyTreebank.Sentence> train,
            Budget budget, long seed) {
        SplittableRandom init = new SplittableRandom(seed + 3);
        int tags = 2 * OFFSET + 1;
        int vocab = ToyTreebank.VOCABULARY.size();
        Var embed = Var.param(Tensor.uniform(Shape.of(vocab, 32), -0.5f, 0.5f, init));
        Layer lstm = Recurrent.lstm(48).returnSequences(true).bidirectional(Recurrent.Merge.CONCAT)
                .create(Shape.of(1, 32), init);
        float limit = (float) Math.sqrt(6.0 / (96 + tags));
        Var out = Var.param(Tensor.uniform(Shape.of(96, tags), -limit, limit, init));
        LinearChainCrf crf = new LinearChainCrf(tags, init);
        List<Var> params = new ArrayList<>(List.of(embed, out));
        params.addAll(lstm.parameters());
        params.addAll(crf.parameters());
        Function<List<ToyTreebank.Sentence>, Var> emissions = batch -> {
            int n = batch.get(0).length();
            float[] ids = new float[batch.size() * n];
            for (int b = 0; b < batch.size(); b++) {
                for (int i = 0; i < n; i++) {
                    ids[b * n + i] = batch.get(b).words()[i];
                }
            }
            return lstm.forward(embed.gather(0, Tensor.of(Shape.of(batch.size(), n), ids))).matmul(out);
        };
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 9);
        for (int s = 0; s < budget.steps(); s++) {
            List<ToyTreebank.Sentence> batch = new ArrayList<>();
            for (int i : sameLength(train, budget.batch(), r)) {
                batch.add(train.get(i));
            }
            int n = batch.get(0).length();
            float[] gold = new float[batch.size() * n];
            for (int b = 0; b < batch.size(); b++) {
                for (int i = 0; i < n; i++) {
                    int h = batch.get(b).heads()[i];
                    gold[b * n + i] = h == 0 ? OFFSET : Math.clamp(h - (i + 1), -OFFSET, OFFSET) + OFFSET;
                }
            }
            Var loss = crf.negativeLogLikelihood(emissions.apply(batch), Tensor.of(Shape.of(batch.size(), n), gold));
            loss.backward();
            opt.step(params);
        }
        return s -> {
            int[] tagsOut = Var.noGrad(() -> crf.decode(emissions.apply(List.of(s)).value()))[0];
            int[] h = new int[s.length()];
            for (int i = 0; i < h.length; i++) {
                int o = tagsOut[i] - OFFSET;
                h[i] = o == 0 ? 0 : Math.clamp(i + 1 + o, 0, h.length);
            }
            return new int[][] {h, null};
        };
    }

    /**
     * The constituency comparison.
     *
     * @param budget sizes
     * @param seed the seed
     * @return per model {@code {F1, exact match, well-formed, PP attachment, F1 on long, ms}}
     */
    public static Map<String, double[]> constituency(Budget budget, long seed) {
        List<ToyTreebank.Sentence> train = ToyTreebank.sample(budget.train(), 4, budget.maxTrain(), seed);
        List<ToyTreebank.Sentence> test = ToyTreebank.sample(budget.test(), 4, budget.maxTest(), seed + 1);
        int vocab = ToyTreebank.VOCABULARY.size();
        int phrases = ToyTreebank.PHRASES.size();
        List<List<int[]>> gold = train.stream().map(ToyTreebank.Sentence::spans).toList();
        Map<String, double[]> out = new LinkedHashMap<>();
        record Config(String name, ChartParser.Encoder encoder, ChartParser.Objective objective) {
        }
        for (Config c : List.of(new Config("chart BiLSTM margin", ChartParser.Encoder.BILSTM, ChartParser.Objective.MARGIN),
                new Config("chart BiLSTM tree-CRF", ChartParser.Encoder.BILSTM, ChartParser.Objective.TREE_CRF),
                new Config("chart factored attention margin", ChartParser.Encoder.FACTORED_ATTENTION,
                        ChartParser.Objective.MARGIN))) {
            long t0 = System.currentTimeMillis();
            ChartParser p = new ChartParser(vocab, phrases, c.encoder(), 32, 48, 2, 64, seed);
            p.train(words(train), gold, c.objective(), budget.steps(), budget.batch(), 3e-3f, seed);
            long ms = System.currentTimeMillis() - t0;
            out.put(c.name(), scoreConstituents(test, budget.maxTrain(), s -> p.parse(s.words()), ms));
        }
        long t0 = System.currentTimeMillis();
        TreebankPcfg pcfg = new TreebankPcfg(train);
        out.put("treebank PCFG", scoreConstituents(test, budget.maxTrain(), s -> pcfg.parse(s.words()),
                System.currentTimeMillis() - t0));
        out.put("right-branching", scoreConstituents(test, budget.maxTrain(), s -> {
            List<int[]> spans = new ArrayList<>();
            spans.add(new int[] {0, 0, s.length()});
            for (int i = 1; i + 1 < s.length(); i++) {
                spans.add(new int[] {3, i, s.length()});
            }
            return spans;
        }, 0));
        return out;
    }

    private static Set<List<Integer>> set(List<int[]> spans) {
        Set<List<Integer>> s = new HashSet<>();
        for (int[] x : spans) {
            s.add(List.of(x[0], x[1], x[2]));
        }
        return s;
    }

    /** The smallest constituent strictly containing the span {@code (start, end)}. */
    private static List<Integer> container(Set<List<Integer>> spans, int start, int end) {
        List<Integer> best = null;
        for (List<Integer> s : spans) {
            boolean contains = s.get(1) <= start && s.get(2) >= end && s.get(2) - s.get(1) > end - start;
            if (contains && (best == null || s.get(2) - s.get(1) < best.get(2) - best.get(1))) {
                best = s;
            }
        }
        return best;
    }

    private static double[] scoreConstituents(List<ToyTreebank.Sentence> test, int maxTrain,
            Function<ToyTreebank.Sentence, List<int[]>> parse, long ms) {
        double matched = 0;
        double predicted = 0;
        double goldCount = 0;
        double longMatched = 0;
        double longPredicted = 0;
        double longGold = 0;
        double exact = 0;
        double formed = 0;
        double pp = 0;
        double ppTotal = 0;
        int pIndex = ToyTreebank.PHRASES.indexOf("PP");
        for (ToyTreebank.Sentence s : test) {
            List<int[]> raw = parse.apply(s);
            formed += raw == null ? 0 : 1;
            Set<List<Integer>> p = raw == null ? Set.of() : set(raw);
            Set<List<Integer>> g = set(s.spans());
            Set<List<Integer>> both = new HashSet<>(p);
            both.retainAll(g);
            matched += both.size();
            predicted += p.size();
            goldCount += g.size();
            if (s.length() > maxTrain) {
                longMatched += both.size();
                longPredicted += p.size();
                longGold += g.size();
            }
            exact += p.equals(g) ? 1 : 0;
            for (int c : s.ppCases()) {
                for (List<Integer> span : g) {
                    if (span.get(0) == pIndex && span.get(1) == c) {
                        ppTotal++;
                        List<Integer> want = container(g, c, span.get(2));
                        pp += want != null && want.equals(container(p, c, span.get(2))) ? 1 : 0;
                    }
                }
            }
        }
        return new double[] {f1(matched, predicted, goldCount), exact / test.size(), formed / test.size(),
            pp / ppTotal, longGold == 0 ? Double.NaN : f1(longMatched, longPredicted, longGold), ms};
    }

    private static double f1(double matched, double predicted, double gold) {
        double p = predicted == 0 ? 0 : matched / predicted;
        double r = matched / gold;
        return p + r == 0 ? 0 : 2 * p * r / (p + r);
    }

    /**
     * Full budgets: 2000 training sentences of length 4-20, 300 test sentences up to length 30.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        Budget budget = new Budget(2000, 300, 20, 30, 3000, 16);
        print("dependency UAS, LAS, trees, PP, UAS long, ms", dependency(budget, 1));
        print("constituency F1, exact, well-formed, PP, F1 long, ms", constituency(budget, 1));
    }

    private static void print(String title, Map<String, double[]> r) {
        StringBuilder s = new StringBuilder(title).append(':');
        r.forEach((k, v) -> {
            s.append("\n  ").append(k);
            for (double x : v) {
                s.append(String.format(" %.3f", x));
            }
        });
        System.out.println(s);
    }
}
