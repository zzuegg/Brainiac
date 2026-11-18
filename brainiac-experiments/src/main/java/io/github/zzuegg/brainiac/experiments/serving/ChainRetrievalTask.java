package io.github.zzuegg.brainiac.experiments.serving;

import io.github.zzuegg.brainiac.experimental.serving.PerplexityPromptCompressor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A multi-hop key-value retrieval task and an attention model that solves it, for comparing
 * KV-cache eviction and prompt compression. The context is a run of two-token slots: a pair
 * {@code K_a V_b} ("key a points to b"), or filler. Three hops are planted, {@code K_c0 V_c1},
 * {@code K_c1 V_c2}, {@code K_c2 V_c3}, among distractor pairs, and the prompt ends with the
 * question {@code ? K_c0 =}; the answer is {@code V_c1 V_c2 V_c3}.
 *
 * <p>The {@link Reader} is one induction head, the mechanism trained LMs use for this: the key
 * of token {@code t} is the embedding of the token before it, its value the embedding of the
 * token itself, and a value token {@code V_b} queries with the embedding of {@code K_b}. So
 * reading the cache answers hop after hop, and a reader of a <em>compressed</em> prompt
 * recomputes keys from the compressed text (a dropped key token breaks its pair). A wide
 * reader (64 dimensions) is the target; a narrow one (16 dimensions, more crosstalk between
 * symbols) is the draft model. Only the second and third hops' keys are invisible to the
 * question at the end of the prompt: that is what draft-guided selection is meant to fix.
 */
public final class ChainRetrievalTask {

    /** Number of key (and value) symbols. */
    public static final int SYMBOLS = 40;
    /** The filler token. */
    public static final int FILL = 2 * SYMBOLS;
    /** The question mark token. */
    public static final int ASK = FILL + 1;
    /** The equals token. */
    public static final int EQUALS = FILL + 2;
    /** The vocabulary size. */
    public static final int VOCAB = FILL + 3;

    private ChainRetrievalTask() {
    }

    /**
     * One question.
     *
     * @param prompt the context followed by {@code ? K_c0 =}
     * @param answer the value tokens of the hops
     * @param start the question's key token
     */
    public record Problem(int[] prompt, int[] answer, int start) {
    }

    /**
     * Draws a problem.
     *
     * @param length the prompt length (context plus the three question tokens)
     * @param hops the hops
     * @param distractors the distractor pairs
     * @param random the random source
     * @return the problem
     */
    public static Problem problem(int length, int hops, int distractors, SplittableRandom random) {
        int slots = (length - 3) / 2;
        int[] symbols = shuffled(SYMBOLS, random);
        int[] context = new int[2 * slots];
        Arrays.fill(context, FILL);
        int[] where = shuffled(slots, random);
        int[] answer = new int[hops];
        for (int j = 0; j < hops; j++) {
            context[2 * where[j]] = symbols[j];
            context[2 * where[j] + 1] = SYMBOLS + symbols[j + 1];
            answer[j] = SYMBOLS + symbols[j + 1];
        }
        for (int i = 0; i < distractors && hops + i < slots; i++) {
            int a = symbols[hops + 1 + random.nextInt(SYMBOLS - hops - 1)];
            context[2 * where[hops + i]] = a;
            context[2 * where[hops + i] + 1] = SYMBOLS + random.nextInt(SYMBOLS);
        }
        int[] prompt = Arrays.copyOf(context, length);
        int at = 2 * slots;
        while (at < length - 3) {
            prompt[at++] = FILL;
        }
        prompt[length - 3] = ASK;
        prompt[length - 2] = symbols[0];
        prompt[length - 1] = EQUALS;
        return new Problem(prompt, answer, symbols[0]);
    }

    static int[] shuffled(int n, SplittableRandom random) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }

    /** An induction-head reader: target or draft model. */
    public static final class Reader {
        private final float[][] embed;
        private final float beta;

        /**
         * @param dim the embedding width
         * @param beta the attention sharpness
         * @param seed the seed
         */
        public Reader(int dim, float beta, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            embed = new float[VOCAB][dim];
            for (float[] e : embed) {
                double n = 0;
                for (int c = 0; c < dim; c++) {
                    e[c] = (float) r.nextGaussian();
                    n += e[c] * e[c];
                }
                for (int c = 0; c < dim; c++) {
                    e[c] /= (float) Math.sqrt(n);
                }
            }
            this.beta = beta;
        }

        /** @return the embedding width */
        public int dim() {
            return embed[0].length;
        }

        /**
         * @param tokens a token sequence
         * @return the key of each token: the embedding of the token before it
         */
        public float[][] keys(int[] tokens) {
            float[][] k = new float[tokens.length][];
            for (int t = 0; t < tokens.length; t++) {
                k[t] = t == 0 ? new float[dim()] : embed[tokens[t - 1]].clone();
            }
            return k;
        }

        /**
         * @param tokens a token sequence
         * @return the value of each token: its embedding
         */
        public float[][] values(int[] tokens) {
            float[][] v = new float[tokens.length][];
            for (int t = 0; t < tokens.length; t++) {
                v[t] = embed[tokens[t]].clone();
            }
            return v;
        }

        /**
         * @param token a token
         * @return its query, scaled: a value token {@code V_b} asks for {@code K_b}
         */
        public float[] query(int token) {
            int target = token >= SYMBOLS && token < FILL ? token - SYMBOLS : token;
            float[] q = embed[target].clone();
            for (int c = 0; c < q.length; c++) {
                q[c] *= beta;
            }
            return q;
        }

        /**
         * Softmax attention of a query over keys.
         *
         * @param q the query
         * @param keys the keys
         * @param count how many keys (from the start) it may see
         * @return the weights, length {@code keys.length}, zero past {@code count}
         */
        public float[] attend(float[] q, float[][] keys, int count) {
            float[] w = new float[keys.length];
            double max = Double.NEGATIVE_INFINITY;
            double[] s = new double[count];
            for (int j = 0; j < count; j++) {
                double d = 0;
                for (int c = 0; c < q.length; c++) {
                    d += q[c] * keys[j][c];
                }
                s[j] = d;
                max = Math.max(max, d);
            }
            double z = 0;
            for (int j = 0; j < count; j++) {
                s[j] = Math.exp(s[j] - max);
                z += s[j];
            }
            for (int j = 0; j < count; j++) {
                w[j] = (float) (s[j] / z);
            }
            return w;
        }

        /**
         * Decodes the hops from a KV set.
         *
         * @param keys the cached keys
         * @param values the cached values
         * @param start the question's key token
         * @param hops the hops
         * @return the value tokens produced
         */
        public int[] decode(float[][] keys, float[][] values, int start, int hops) {
            int[] out = new int[hops];
            int token = start;
            for (int h = 0; h < hops; h++) {
                float[] w = attend(query(token), keys, keys.length);
                float[] y = new float[dim()];
                for (int j = 0; j < keys.length; j++) {
                    for (int c = 0; c < y.length; c++) {
                        y[c] += w[j] * values[j][c];
                    }
                }
                int best = SYMBOLS;
                double bestScore = Double.NEGATIVE_INFINITY;
                for (int v = SYMBOLS; v < FILL; v++) {
                    double d = 0;
                    for (int c = 0; c < y.length; c++) {
                        d += y[c] * embed[v][c];
                    }
                    if (d > bestScore) {
                        bestScore = d;
                        best = v;
                    }
                }
                out[h] = best;
                token = best;
            }
            return out;
        }

        /**
         * Reads a (possibly compressed) prompt and decodes.
         *
         * @param prompt the prompt it reads
         * @param start the question's key token
         * @param hops the hops
         * @return the value tokens produced
         */
        public int[] read(int[] prompt, int start, int hops) {
            return decode(keys(prompt), values(prompt), start, hops);
        }
    }

    /**
     * The small LM of the prompt compressors: an interpolated bigram plus a cache of the
     * tokens already seen in the sequence, counted on sample problems.
     */
    public static final class CacheBigramLm implements PerplexityPromptCompressor.ScoringLm {
        private final double[][] bigram = new double[VOCAB][VOCAB];

        /**
         * @param samples training problems
         */
        public CacheBigramLm(List<Problem> samples) {
            for (double[] row : bigram) {
                Arrays.fill(row, 0.1);
            }
            for (Problem p : samples) {
                for (int t = 1; t < p.prompt().length; t++) {
                    bigram[p.prompt()[t - 1]][p.prompt()[t]]++;
                }
            }
            for (double[] row : bigram) {
                double s = Arrays.stream(row).sum();
                for (int i = 0; i < row.length; i++) {
                    row[i] /= s;
                }
            }
        }

        @Override
        public double[] logProbs(int[] tokens) {
            double[] lp = new double[tokens.length];
            int[] counts = new int[VOCAB];
            for (int t = 0; t < tokens.length; t++) {
                double cache = t == 0 ? 1.0 / VOCAB : counts[tokens[t]] / (double) t;
                double big = t == 0 ? 1.0 / VOCAB : bigram[tokens[t - 1]][tokens[t]];
                lp[t] = Math.log(0.6 * big + 0.3 * cache + 0.1 / VOCAB);
                counts[tokens[t]]++;
            }
            return lp;
        }
    }

    /**
     * Splits a prompt's context (everything before the question) into documents.
     *
     * @param prompt the prompt
     * @param size tokens per document, even
     * @return the documents
     */
    public static List<int[]> documents(int[] prompt, int size) {
        List<int[]> docs = new ArrayList<>();
        int end = prompt.length - 3;
        for (int s = 0; s < end; s += size) {
            docs.add(Arrays.copyOfRange(prompt, s, Math.min(end, s + size)));
        }
        return docs;
    }
}
