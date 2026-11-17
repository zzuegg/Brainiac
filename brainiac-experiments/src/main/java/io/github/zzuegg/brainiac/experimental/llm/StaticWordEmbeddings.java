package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Static word vectors learned from a tokenised corpus, in three classic flavours:
 *
 * <ul>
 *   <li>{@link Kind#SGNS}: word2vec skip-gram with negative sampling. Each (centre, context)
 *       pair within the window raises {@code sigma(v_c . u_o)}, and {@code k} words drawn from
 *       the unigram distribution to the 3/4 lower it.</li>
 *   <li>{@link Kind#GLOVE}: weighted least squares on the log co-occurrence counts,
 *       {@code sum f(X_ij) (w_i . w~_j + b_i + b~_j - log X_ij)^2} with
 *       {@code f(x) = min(1, (x / x_max)^0.75)}, trained with AdaGrad; the vector is
 *       {@code w + w~}.</li>
 *   <li>{@link Kind#FASTTEXT}: SGNS whose centre vector is the mean of the word's own vector and
 *       of hashed character n-gram vectors ({@code n = 3..5} of {@code <word>}), so unseen words
 *       get vectors from their pieces.</li>
 * </ul>
 *
 * The trainers are plain loops (the corpus is small and the updates sparse);
 * {@link #table()} hands the vectors to core {@code Embedding.pretrained}.
 *
 * <p>Papers: Mikolov et al., "Distributed Representations of Words and Phrases and their
 * Compositionality" (NeurIPS 2013), <a href="https://arxiv.org/abs/1310.4546">arXiv:1310.4546</a>;
 * Pennington et al., "GloVe" (EMNLP 2014); Bojanowski et al., "Enriching Word Vectors with
 * Subword Information" (TACL 2017), <a href="https://arxiv.org/abs/1607.04606">arXiv:1607.04606</a>.
 */
public final class StaticWordEmbeddings {

    /** The training objective. */
    public enum Kind {
        /** skip-gram with negative sampling */
        SGNS,
        /** global vectors */
        GLOVE,
        /** skip-gram over character n-grams */
        FASTTEXT
    }

    private static final int MIN_N = 3;
    private static final int MAX_N = 5;
    private static final int BUCKETS = 2048;
    private static final int NEGATIVES = 5;

    private final Kind kind;
    private final List<String> words;
    private final Map<String, Integer> ids = new HashMap<>();
    private final int dim;
    private final float[][] vectors;
    private final float[][] grams;

    private StaticWordEmbeddings(Kind kind, List<String> words, int dim, float[][] vectors, float[][] grams) {
        this.kind = kind;
        this.words = List.copyOf(words);
        for (int i = 0; i < words.size(); i++) {
            ids.put(words.get(i), i);
        }
        this.dim = dim;
        this.vectors = vectors;
        this.grams = grams;
    }

    /**
     * Trains word vectors.
     *
     * @param kind the objective
     * @param vocabulary the words, indexed by id
     * @param sentences token ids
     * @param dim the vector size
     * @param window the context window on each side
     * @param epochs passes over the corpus
     * @param seed the seed
     * @return the trained vectors
     */
    public static StaticWordEmbeddings train(Kind kind, List<String> vocabulary, int[][] sentences, int dim,
            int window, int epochs, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int v = vocabulary.size();
        if (kind == Kind.GLOVE) {
            return glove(vocabulary, sentences, dim, window, epochs, r);
        }
        float[][] in = init(v, dim, r, 0.5f / dim);
        float[][] out = new float[v][dim];
        float[][] gram = kind == Kind.FASTTEXT ? init(BUCKETS, dim, r, 0.5f / dim) : null;
        int[][] pieces = new int[v][];
        for (int w = 0; w < v; w++) {
            pieces[w] = kind == Kind.FASTTEXT ? ngrams(vocabulary.get(w)) : new int[0];
        }
        int[] table = unigramTable(vocabulary.size(), sentences);
        long total = (long) epochs * sentences.length;
        long done = 0;
        float[] centre = new float[dim];
        float[] grad = new float[dim];
        for (int e = 0; e < epochs; e++) {
            for (int[] s : sentences) {
                float lr = 0.05f * Math.max(0.05f, 1 - (float) done++ / total);
                for (int i = 0; i < s.length; i++) {
                    int c = s[i];
                    int[] g = pieces[c];
                    float scale = 1f / (1 + g.length);
                    for (int i2 = Math.max(0, i - window); i2 <= Math.min(s.length - 1, i + window); i2++) {
                        if (i2 == i) {
                            continue;
                        }
                        System.arraycopy(in[c], 0, centre, 0, dim);
                        for (int p : g) {
                            for (int d = 0; d < dim; d++) {
                                centre[d] += gram[p][d];
                            }
                        }
                        for (int d = 0; d < dim; d++) {
                            centre[d] *= scale;
                            grad[d] = 0;
                        }
                        for (int k = 0; k <= NEGATIVES; k++) {
                            int o = k == 0 ? s[i2] : table[r.nextInt(table.length)];
                            if (k > 0 && o == s[i2]) {
                                continue;
                            }
                            float dot = 0;
                            for (int d = 0; d < dim; d++) {
                                dot += centre[d] * out[o][d];
                            }
                            float step = lr * ((k == 0 ? 1 : 0) - sigmoid(dot));
                            for (int d = 0; d < dim; d++) {
                                grad[d] += step * out[o][d];
                                out[o][d] += step * centre[d];
                            }
                        }
                        for (int d = 0; d < dim; d++) {
                            in[c][d] += grad[d] * scale;
                        }
                        for (int p : g) {
                            for (int d = 0; d < dim; d++) {
                                gram[p][d] += grad[d] * scale;
                            }
                        }
                    }
                }
            }
        }
        return new StaticWordEmbeddings(kind, vocabulary, dim, in, gram);
    }

    private static StaticWordEmbeddings glove(List<String> vocabulary, int[][] sentences, int dim, int window,
            int epochs, SplittableRandom r) {
        int v = vocabulary.size();
        double[][] x = new double[v][v];
        for (int[] s : sentences) {
            for (int i = 0; i < s.length; i++) {
                for (int j = Math.max(0, i - window); j < i; j++) {
                    double w = 1.0 / (i - j);
                    x[s[i]][s[j]] += w;
                    x[s[j]][s[i]] += w;
                }
            }
        }
        float[][] w = init(v, dim, r, 0.5f / dim);
        float[][] wt = init(v, dim, r, 0.5f / dim);
        float[] b = new float[v];
        float[] bt = new float[v];
        float[][] gw = fill(v, dim);
        float[][] gwt = fill(v, dim);
        float[] gb = new float[v];
        float[] gbt = new float[v];
        java.util.Arrays.fill(gb, 1);
        java.util.Arrays.fill(gbt, 1);
        double xmax = 100;
        float lr = 0.05f;
        List<int[]> nonzero = new ArrayList<>();
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                if (x[i][j] > 0) {
                    nonzero.add(new int[] {i, j});
                }
            }
        }
        for (int e = 0; e < epochs; e++) {
            for (int n = 0; n < nonzero.size(); n++) {
                int[] p = nonzero.get(r.nextInt(nonzero.size()));
                int i = p[0];
                int j = p[1];
                double dot = b[i] + bt[j] - Math.log(x[i][j]);
                for (int d = 0; d < dim; d++) {
                    dot += w[i][d] * wt[j][d];
                }
                float f = (float) Math.min(1, Math.pow(x[i][j] / xmax, 0.75));
                float g = (float) (f * dot);
                for (int d = 0; d < dim; d++) {
                    float g1 = g * wt[j][d];
                    float g2 = g * w[i][d];
                    w[i][d] -= lr * g1 / (float) Math.sqrt(gw[i][d]);
                    wt[j][d] -= lr * g2 / (float) Math.sqrt(gwt[j][d]);
                    gw[i][d] += g1 * g1;
                    gwt[j][d] += g2 * g2;
                }
                b[i] -= lr * g / (float) Math.sqrt(gb[i]);
                bt[j] -= lr * g / (float) Math.sqrt(gbt[j]);
                gb[i] += g * g;
                gbt[j] += g * g;
            }
        }
        for (int i = 0; i < v; i++) {
            for (int d = 0; d < dim; d++) {
                w[i][d] += wt[i][d];
            }
        }
        return new StaticWordEmbeddings(Kind.GLOVE, vocabulary, dim, w, null);
    }

    private static float[][] init(int rows, int dim, SplittableRandom r, float scale) {
        float[][] m = new float[rows][dim];
        for (float[] row : m) {
            for (int d = 0; d < dim; d++) {
                row[d] = (float) ((r.nextDouble() * 2 - 1) * scale * dim);
            }
        }
        return m;
    }

    private static float[][] fill(int rows, int dim) {
        float[][] m = new float[rows][dim];
        for (float[] row : m) {
            java.util.Arrays.fill(row, 1f);
        }
        return m;
    }

    private static int[] unigramTable(int v, int[][] sentences) {
        double[] count = new double[v];
        for (int[] s : sentences) {
            for (int t : s) {
                count[t]++;
            }
        }
        double z = 0;
        for (int i = 0; i < v; i++) {
            count[i] = Math.pow(count[i], 0.75);
            z += count[i];
        }
        int size = 100 * v;
        int[] table = new int[size];
        int w = 0;
        double acc = count[0] / z;
        for (int i = 0; i < size; i++) {
            table[i] = w;
            if ((i + 1.0) / size > acc && w < v - 1) {
                w++;
                acc += count[w] / z;
            }
        }
        return table;
    }

    /**
     * The hashed character n-gram buckets of a word, {@code n = 3..5} over {@code <word>}.
     *
     * @param word the word
     * @return bucket ids
     */
    public static int[] ngrams(String word) {
        String s = "<" + word + ">";
        List<Integer> out = new ArrayList<>();
        for (int n = MIN_N; n <= MAX_N; n++) {
            for (int i = 0; i + n <= s.length(); i++) {
                int h = 0x811c9dc5;
                for (int k = i; k < i + n; k++) {
                    h = (h ^ s.charAt(k)) * 0x01000193;
                }
                out.add(Math.floorMod(h, BUCKETS));
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private static float sigmoid(float x) {
        return x > 8 ? 1 : x < -8 ? 0 : (float) (1 / (1 + Math.exp(-x)));
    }

    /** @return the objective these vectors were trained with */
    public Kind kind() {
        return kind;
    }

    /** @return the vector size */
    public int dimension() {
        return dim;
    }

    /**
     * The vector of a word. Out-of-vocabulary words get the mean of their n-gram vectors under
     * fastText and a zero vector otherwise.
     *
     * @param word the word
     * @return a fresh vector
     */
    public float[] vector(String word) {
        Integer id = ids.get(word);
        float[] v = new float[dim];
        int parts = 0;
        if (id != null) {
            System.arraycopy(vectors[id], 0, v, 0, dim);
            parts++;
        }
        if (grams != null) {
            for (int p : ngrams(word)) {
                for (int d = 0; d < dim; d++) {
                    v[d] += grams[p][d];
                }
                parts++;
            }
        }
        if (parts > 1) {
            for (int d = 0; d < dim; d++) {
                v[d] /= parts;
            }
        }
        return v;
    }

    /** @return {@code [vocabulary, dimension]} for {@code Embedding.pretrained} */
    public Tensor table() {
        float[] t = new float[words.size() * dim];
        for (int w = 0; w < words.size(); w++) {
            System.arraycopy(vector(words.get(w)), 0, t, w * dim, dim);
        }
        return Tensor.of(Shape.of(words.size(), dim), t);
    }

    /**
     * Cosine similarity.
     *
     * @param a a vector
     * @param b a vector
     * @return {@code a . b / (|a| |b|)}, 0 for a zero vector
     */
    public static double cosine(float[] a, float[] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int d = 0; d < a.length; d++) {
            dot += a[d] * b[d];
            na += a[d] * a[d];
            nb += b[d] * b[d];
        }
        return na == 0 || nb == 0 ? 0 : dot / Math.sqrt(na * nb);
    }

    /**
     * Solves {@code a : b :: c : ?} by 3CosAdd: the vocabulary word nearest to
     * {@code b - a + c} (by cosine, on unit-normalised vectors), excluding the three inputs.
     *
     * @param a the first word
     * @param b its counterpart
     * @param c the query word
     * @return the answer
     */
    public String analogy(String a, String b, String c) {
        float[] va = unit(vector(a));
        float[] vb = unit(vector(b));
        float[] vc = unit(vector(c));
        float[] target = new float[dim];
        for (int d = 0; d < dim; d++) {
            target[d] = vb[d] - va[d] + vc[d];
        }
        String best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (String w : words) {
            if (w.equals(a) || w.equals(b) || w.equals(c)) {
                continue;
            }
            double s = cosine(target, vector(w));
            if (s > bestScore) {
                bestScore = s;
                best = w;
            }
        }
        return best;
    }

    private static float[] unit(float[] v) {
        double n = 0;
        for (float x : v) {
            n += x * x;
        }
        float[] u = v.clone();
        if (n > 0) {
            for (int d = 0; d < u.length; d++) {
                u[d] /= (float) Math.sqrt(n);
            }
        }
        return u;
    }

    @Override
    public String toString() {
        return "StaticWordEmbeddings(" + kind + ", " + words.size() + " words x " + dim + ")";
    }
}
