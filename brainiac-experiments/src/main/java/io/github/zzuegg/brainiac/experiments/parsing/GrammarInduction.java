package io.github.zzuegg.brainiac.experiments.parsing;

import io.github.zzuegg.brainiac.experimental.data.ToyTreebank;
import io.github.zzuegg.brainiac.experimental.parse.NeuralPcfg;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 11 item 11.5.10: grammar induction on {@link ToyTreebank} sentences. Unlabelled
 * sentence-level span F1 (trivial spans, the whole sentence and single words, excluded) of
 * flat, neural and compound PCFGs (Viterbi and MBR decoding) trained on raw words only, against
 * right-branching, left-branching and random binary trees, with the supervised
 * {@link TreebankPcfg} as the ceiling.
 */
public final class GrammarInduction {

    private GrammarInduction() {
    }

    /**
     * A row.
     *
     * @param method the method
     * @param f1 mean sentence F1
     * @param nll the per-word negative log-likelihood on test ({@code NaN} for baselines)
     */
    public record Row(String method, double f1, double nll) {
    }

    private static Set<Long> nonTrivial(List<int[]> spans, int n, int startIndex) {
        Set<Long> s = new HashSet<>();
        for (int[] x : spans) {
            int a = x[startIndex];
            int b = x[startIndex + 1];
            if (b - a >= 2 && b - a < n) {
                s.add(((long) a << 32) | b);
            }
        }
        return s;
    }

    private static double f1(Set<Long> pred, Set<Long> gold) {
        if (gold.isEmpty() && pred.isEmpty()) {
            return 1;
        }
        int hit = 0;
        for (long p : pred) {
            hit += gold.contains(p) ? 1 : 0;
        }
        return 2.0 * hit / Math.max(1, pred.size() + gold.size());
    }

    private static double score(List<ToyTreebank.Sentence> test, Function<int[], List<int[]>> parser, int start) {
        double s = 0;
        for (ToyTreebank.Sentence t : test) {
            List<int[]> p = parser.apply(t.words());
            Set<Long> pred = p == null ? Set.of() : nonTrivial(p, t.length(), start);
            s += f1(pred, nonTrivial(t.spans(), t.length(), 1));
        }
        return s / test.size();
    }

    private static List<int[]> branching(int n, boolean right) {
        List<int[]> s = new ArrayList<>();
        for (int w = n; w >= 2; w--) {
            s.add(right ? new int[] {n - w, n} : new int[] {0, w});
        }
        return s;
    }

    private static void random(int i, int j, SplittableRandom r, List<int[]> out) {
        if (j - i < 2) {
            return;
        }
        out.add(new int[] {i, j});
        int k = i + 1 + r.nextInt(j - i - 1);
        random(i, k, r, out);
        random(k, j, r, out);
    }

    /**
     * Runs the comparison.
     *
     * @param train training sentences
     * @param test test sentences
     * @param steps training steps per grammar
     * @param seed the seed
     * @return rows
     */
    public static List<Row> run(int train, int test, int steps, long seed) {
        List<ToyTreebank.Sentence> tr = ToyTreebank.sample(train, 3, 8, seed);
        List<ToyTreebank.Sentence> te = ToyTreebank.sample(test, 3, 8, seed + 1);
        List<int[]> words = new ArrayList<>();
        for (ToyTreebank.Sentence s : tr) {
            words.add(s.words());
        }
        int vocab = ToyTreebank.VOCABULARY.size();
        List<Row> rows = new ArrayList<>();
        rows.add(new Row("right-branching", score(te, w -> branching(w.length, true), 0), Double.NaN));
        rows.add(new Row("left-branching", score(te, w -> branching(w.length, false), 0), Double.NaN));
        SplittableRandom rr = new SplittableRandom(seed);
        rows.add(new Row("random trees", score(te, w -> {
            List<int[]> o = new ArrayList<>();
            random(0, w.length, rr, o);
            return o;
        }, 0), Double.NaN));
        for (NeuralPcfg.Parametrisation p : NeuralPcfg.Parametrisation.values()) {
            NeuralPcfg g = new NeuralPcfg(p, 6, 12, vocab, 24, 4, seed);
            g.train(words, steps, 12, p == NeuralPcfg.Parametrisation.FLAT ? 5e-2f : 1e-2f, seed);
            double nll = 0;
            int count = 0;
            for (ToyTreebank.Sentence s : te) {
                nll -= g.logLikelihood(s.words());
                count += s.length();
            }
            for (NeuralPcfg.Decoding d : NeuralPcfg.Decoding.values()) {
                rows.add(new Row(p + " PCFG " + d, score(te, w -> g.parse(w, d), 0), nll / count));
            }
        }
        TreebankPcfg sup = new TreebankPcfg(tr);
        rows.add(new Row("treebank PCFG (supervised)", score(te, sup::parse, 1), Double.NaN));
        return rows;
    }
}
