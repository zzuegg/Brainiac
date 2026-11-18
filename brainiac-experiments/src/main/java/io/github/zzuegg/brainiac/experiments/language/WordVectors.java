package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.StaticWordEmbeddings;
import io.github.zzuegg.brainiac.experimental.llm.StaticWordEmbeddings.Kind;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Embedding;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 6 item 2.14: word2vec (SGNS), GloVe and fastText on the {@link SyntheticGrammar}
 * corpus. Reported per method:
 *
 * <ul>
 *   <li>analogy accuracy on the planted gender ({@code dako : daka :: miro : mira}) and number
 *       ({@code dako : dakos :: miro : miros}) analogies, by 3CosAdd over the whole vocabulary;</li>
 *   <li>similarity: Spearman correlation between cosine and a gold similarity (shared stem,
 *       animacy, gender and number) over all pairs of nouns;</li>
 *   <li>OOV robustness: the held-out feminine plurals and every form of 8 unseen stems are
 *       classified into gender x number by the nearest form centroid of the in-vocabulary
 *       nouns (word-level methods have no vector for them: chance is 0.25);</li>
 *   <li>downstream: a noun-form classifier (core {@code Embedding} plus a dense layer) trained
 *       on the nouns of 4 stems and tested on the others, warm-started with
 *       {@code Embedding.pretrained} against random initialisation.</li>
 * </ul>
 */
public final class WordVectors {

    private static final String[] NEW_STEMS = {"qex", "wib", "hul", "jor", "nep", "yat", "cim", "dru"};
    private static final int[] LABELED_STEMS = {0, 1, 6, 7};

    private WordVectors() {
    }

    /**
     * The corpus every method trains on.
     *
     * @param g the grammar
     * @param seed the seed
     * @return 20000 sentences of ids
     */
    public static int[][] corpus(SyntheticGrammar g, long seed) {
        return g.sentences(20000, new SplittableRandom(seed))[0];
    }

    /**
     * Trains one method.
     *
     * @param kind the method
     * @param g the grammar
     * @param seed the seed
     * @return the vectors (dimension 24, window 4: the whole sentence)
     */
    public static StaticWordEmbeddings train(Kind kind, SyntheticGrammar g, long seed) {
        return StaticWordEmbeddings.train(kind, g.words(), corpus(g, seed), 24, 4, kind == Kind.GLOVE ? 300 : 10, seed);
    }

    /**
     * Analogy accuracy over every in-vocabulary gender and number analogy.
     *
     * @param e the vectors
     * @param g the grammar
     * @return the fraction answered exactly
     */
    public static double analogies(StaticWordEmbeddings e, SyntheticGrammar g) {
        int hit = 0;
        int total = 0;
        for (int flip : new int[] {1, 2}) {
            for (int s1 = 0; s1 < SyntheticGrammar.stems(); s1++) {
                for (int s2 = 0; s2 < SyntheticGrammar.stems(); s2++) {
                    if (s1 == s2) {
                        continue;
                    }
                    for (int f = 0; f < 4; f++) {
                        String a = SyntheticGrammar.noun(s1, f);
                        String b = SyntheticGrammar.noun(s1, f ^ flip);
                        String c = SyntheticGrammar.noun(s2, f);
                        String d = SyntheticGrammar.noun(s2, f ^ flip);
                        if (g.id(a) < 0 || g.id(b) < 0 || g.id(c) < 0 || g.id(d) < 0 || (f & flip) != 0) {
                            continue;
                        }
                        total++;
                        hit += d.equals(e.analogy(a, b, c)) ? 1 : 0;
                    }
                }
            }
        }
        return (double) hit / total;
    }

    /**
     * Spearman correlation between cosine similarity and the gold similarity over noun pairs.
     *
     * @param e the vectors
     * @param g the grammar
     * @return the rank correlation
     */
    public static double similarity(StaticWordEmbeddings e, SyntheticGrammar g) {
        List<int[]> nouns = new ArrayList<>();
        for (int s = 0; s < SyntheticGrammar.stems(); s++) {
            for (int f = 0; f < 4; f++) {
                if (g.id(SyntheticGrammar.noun(s, f)) >= 0) {
                    nouns.add(new int[] {s, f});
                }
            }
        }
        List<double[]> pairs = new ArrayList<>();
        for (int i = 0; i < nouns.size(); i++) {
            for (int j = i + 1; j < nouns.size(); j++) {
                int[] a = nouns.get(i);
                int[] b = nouns.get(j);
                double gold = (a[0] == b[0] ? 2 : 0) + (SyntheticGrammar.animate(a[0]) == SyntheticGrammar.animate(b[0])
                        ? 1 : 0) + ((a[1] & 1) == (b[1] & 1) ? 1 : 0) + ((a[1] & 2) == (b[1] & 2) ? 1 : 0);
                double cos = StaticWordEmbeddings.cosine(e.vector(SyntheticGrammar.noun(a[0], a[1])),
                        e.vector(SyntheticGrammar.noun(b[0], b[1])));
                pairs.add(new double[] {gold, cos});
            }
        }
        double[] x = ranks(pairs, 0);
        double[] y = ranks(pairs, 1);
        return pearson(x, y);
    }

    private static double[] ranks(List<double[]> pairs, int column) {
        int n = pairs.size();
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(pairs.get(a)[column], pairs.get(b)[column]));
        double[] r = new double[n];
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && pairs.get(order[j + 1])[column] == pairs.get(order[i])[column]) {
                j++;
            }
            for (int k = i; k <= j; k++) {
                r[order[k]] = (i + j) / 2.0;
            }
            i = j + 1;
        }
        return r;
    }

    private static double pearson(double[] x, double[] y) {
        double mx = 0;
        double my = 0;
        for (int i = 0; i < x.length; i++) {
            mx += x[i] / x.length;
            my += y[i] / y.length;
        }
        double sxy = 0;
        double sxx = 0;
        double syy = 0;
        for (int i = 0; i < x.length; i++) {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx) * (x[i] - mx);
            syy += (y[i] - my) * (y[i] - my);
        }
        return sxy / Math.sqrt(sxx * syy);
    }

    /**
     * Classifies out-of-vocabulary nouns (held-out feminine plurals and unseen stems) into
     * gender x number by the nearest centroid of the in-vocabulary forms.
     *
     * @param e the vectors
     * @param g the grammar
     * @return the accuracy (chance 0.25)
     */
    public static double oov(StaticWordEmbeddings e, SyntheticGrammar g) {
        float[][] centroid = new float[4][e.dimension()];
        for (int s = 0; s < SyntheticGrammar.stems(); s++) {
            for (int f = 0; f < 4; f++) {
                String w = SyntheticGrammar.noun(s, f);
                if (g.id(w) >= 0) {
                    float[] v = e.vector(w);
                    for (int d = 0; d < v.length; d++) {
                        centroid[f][d] += v[d];
                    }
                }
            }
        }
        List<String> words = new ArrayList<>();
        List<Integer> forms = new ArrayList<>();
        for (int s : SyntheticGrammar.heldOutStems()) {
            words.add(SyntheticGrammar.noun(s, 3));
            forms.add(3);
        }
        for (String stem : NEW_STEMS) {
            for (int f = 0; f < 4; f++) {
                words.add(stem + ((f & 1) == 0 ? "o" : "a") + (f >= 2 ? "s" : ""));
                forms.add(f);
            }
        }
        double hit = 0;
        for (int i = 0; i < words.size(); i++) {
            float[] v = e.vector(words.get(i));
            int best = 0;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int f = 0; f < 4; f++) {
                double c = StaticWordEmbeddings.cosine(v, centroid[f]);
                if (c > bestScore) {
                    bestScore = c;
                    best = f;
                }
            }
            boolean zero = true;
            for (float x : v) {
                zero &= x == 0;
            }
            // a word without a vector can only be guessed uniformly
            hit += zero ? 0.25 : best == forms.get(i) ? 1 : 0;
        }
        return hit / words.size();
    }

    /**
     * Trains the downstream noun-form classifier on the nouns of 4 stems and tests it on the
     * nouns of the other stems.
     *
     * @param pretrained vectors to warm-start the embedding with, or {@code null} for random
     * @param g the grammar
     * @param seed the seed
     * @return the test accuracy (chance 0.25)
     */
    public static double downstream(StaticWordEmbeddings pretrained, SyntheticGrammar g, long seed) {
        Embedding.Spec emb = pretrained != null ? Embedding.pretrained(pretrained.table()) : Embedding.of(g.size(), 24);
        Model m = Model.sequential().input(1).add(emb).flatten().dense(4, Activation.IDENTITY).seed(seed).build();
        List<float[]> train = new ArrayList<>();
        List<float[]> test = new ArrayList<>();
        for (int s = 0; s < SyntheticGrammar.stems(); s++) {
            boolean labeled = false;
            for (int l : LABELED_STEMS) {
                labeled |= l == s;
            }
            for (int f = 0; f < 4; f++) {
                int id = g.id(SyntheticGrammar.noun(s, f));
                if (id >= 0) {
                    (labeled ? train : test).add(new float[] {id, f});
                }
            }
        }
        Optimizer opt = Optimizer.adam(1e-2f);
        Tensor[] tr = tensors(train);
        for (int step = 0; step < 150; step++) {
            Functional.crossEntropy(m.forward(Var.of(tr[0]), true), tr[1]).backward();
            opt.step(m.parameters());
        }
        Tensor[] te = tensors(test);
        return m.forward(Var.of(te[0])).value().argmax(1).equalTo(te[1]).mean().item();
    }

    private static Tensor[] tensors(List<float[]> rows) {
        float[] x = new float[rows.size()];
        float[] y = new float[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            x[i] = rows.get(i)[0];
            y[i] = rows.get(i)[1];
        }
        return new Tensor[] {Tensor.of(Shape.of(rows.size(), 1), x), Tensor.of(Shape.of(rows.size()), y)};
    }
}
