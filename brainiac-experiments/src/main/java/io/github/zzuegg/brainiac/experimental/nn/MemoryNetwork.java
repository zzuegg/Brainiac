package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Memory networks for question answering over a story of sentences, answered by soft
 * attention over a sentence memory, repeated for several "hops":
 *
 * <ul>
 *   <li>{@link Kind#END_TO_END} (MemN2N): each sentence is stored twice, as an input memory
 *       {@code m_i = sum_j l_j * A_k x_ij + T_A(i)} and an output memory {@code c_i} (embedded
 *       with {@code A_(k+1)}, adjacent weight tying), where {@code l_j} is the position
 *       encoding and {@code T} a learned recency embedding. Hop {@code k} attends
 *       {@code p = softmax(u . m_i)}, reads {@code o = sum p_i c_i} and updates
 *       {@code u = u + o}; the answer is {@code softmax(W u)}.</li>
 *   <li>{@link Kind#KEY_VALUE} (KV-MemNN): addressing and reading are decoupled. Keys are the
 *       sentence encodings, values a separate embedding of one value token per memory (such as
 *       the sentence's location or object); hops update {@code q = R_k (q + o)}.</li>
 * </ul>
 *
 * <p>Several hops let the model chain supporting facts: find who holds the apple, then where
 * that person is. Stories are {@code [batch][sentences][words]} ids with 0 as padding; an
 * all-padding sentence is an empty memory slot.
 *
 * <p>Papers: Sukhbaatar et al., "End-To-End Memory Networks" (NeurIPS 2015),
 * <a href="https://arxiv.org/abs/1503.08895">arXiv:1503.08895</a>; Miller et al., "Key-Value
 * Memory Networks for Directly Reading Documents" (EMNLP 2016),
 * <a href="https://arxiv.org/abs/1606.03126">arXiv:1606.03126</a>.
 */
public final class MemoryNetwork {

    /** The memory network variant. */
    public enum Kind {
        /** MemN2N with adjacent tying */
        END_TO_END,
        /** key-value memory network */
        KEY_VALUE
    }

    private final Kind kind;
    private final int dim;
    private final int hops;
    private final int sentences;
    private final int words;
    private final List<Var> embeddings = new ArrayList<>();
    private final List<Var> temporal = new ArrayList<>();
    private final Var values;
    private final List<Var> rotations = new ArrayList<>();
    private final Var answer;
    private final Tensor position;
    private float[][] lastAttention;

    private MemoryNetwork(int vocab, int dim, int hops, int sentences, int words, int answers, Kind kind, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.kind = kind;
        this.dim = dim;
        this.hops = hops;
        this.sentences = sentences;
        this.words = words;
        int tables = kind == Kind.END_TO_END ? hops + 1 : 1;
        for (int i = 0; i < tables; i++) {
            embeddings.add(Var.param(Tensor.normal(Shape.of(vocab, dim), 0, 0.1f, r)));
            temporal.add(Var.param(Tensor.normal(Shape.of(sentences, dim), 0, 0.1f, r)));
        }
        if (kind == Kind.KEY_VALUE) {
            values = Var.param(Tensor.normal(Shape.of(vocab, dim), 0, 0.1f, r));
            for (int i = 0; i < hops; i++) {
                rotations.add(Var.param(Tensor.normal(Shape.of(dim, dim), 0, (float) (1 / Math.sqrt(dim)), r)
                        .add(eye(dim))));
            }
        } else {
            values = null;
        }
        answer = ClassicOps.matrix(r, dim, answers);
        float[] l = new float[words * dim];
        for (int j = 1; j <= words; j++) {
            for (int k = 1; k <= dim; k++) {
                l[(j - 1) * dim + k - 1] = (1 - (float) j / words) - ((float) k / dim) * (1 - 2f * j / words);
            }
        }
        position = Tensor.of(Shape.of(words, dim), l);
    }

    private static Tensor eye(int n) {
        float[] f = new float[n * n];
        for (int i = 0; i < n; i++) {
            f[i * n + i] = 1;
        }
        return Tensor.of(Shape.of(n, n), f);
    }

    /**
     * Creates a memory network.
     *
     * @param vocab the vocabulary (id 0 is padding)
     * @param dim the embedding width
     * @param hops the attention hops
     * @param sentences the memory slots (sentences per story)
     * @param words the words per sentence and question
     * @param answers the answer classes
     * @param kind the variant
     * @param seed the initialization seed
     * @return the network
     */
    public static MemoryNetwork of(int vocab, int dim, int hops, int sentences, int words, int answers, Kind kind,
            long seed) {
        return new MemoryNetwork(vocab, dim, hops, sentences, words, answers, kind, seed);
    }

    /** Bag-of-words with position encoding: {@code [b, n, words] ids -> [b, n, dim]}; padding embeds to 0. */
    private Var encode(Var table, int[][][] ids) {
        int b = ids.length;
        int n = ids[0].length;
        float[] f = new float[b * n * words];
        float[] mask = new float[b * n * words];
        for (int i = 0; i < b; i++) {
            for (int s = 0; s < n; s++) {
                for (int w = 0; w < words; w++) {
                    int id = w < ids[i][s].length ? ids[i][s][w] : 0;
                    f[(i * n + s) * words + w] = id;
                    mask[(i * n + s) * words + w] = id == 0 ? 0 : 1;
                }
            }
        }
        Var e = table.gather(0, Tensor.of(Shape.of(b, n, words), f));
        return e.mul(Tensor.of(Shape.of(b, n, words, 1), mask)).mul(position).sum(2, false);
    }

    /**
     * Answer logits.
     *
     * @param stories {@code [batch][sentences][words]} ids, 0 for padding; the last sentence is the newest
     * @param questions {@code [batch][words]} ids
     * @param valueTokens for {@link Kind#KEY_VALUE}, the value token of each memory {@code [batch][sentences]};
     *     ignored otherwise (may be {@code null})
     * @return {@code [batch, answers]}
     */
    public Var logits(int[][][] stories, int[][] questions, int[][] valueTokens) {
        int b = stories.length;
        float[] bias = new float[b * sentences];
        for (int i = 0; i < b; i++) {
            for (int s = 0; s < sentences; s++) {
                boolean empty = true;
                for (int w : stories[i][s]) {
                    empty &= w == 0;
                }
                bias[i * sentences + s] = empty ? -1e9f : 0f;
            }
        }
        Tensor maskBias = Tensor.of(Shape.of(b, sentences), bias);
        int[][][] q = new int[b][1][];
        for (int i = 0; i < b; i++) {
            q[i][0] = questions[i];
        }
        Var u = encode(embeddings.get(0), q).reshape(b, dim);
        lastAttention = new float[hops][];
        Var valueMemory = null;
        if (kind == Kind.KEY_VALUE) {
            valueMemory = values.gather(0, ClassicOps.ids(valueTokens));
        }
        Var keyMemory = kind == Kind.KEY_VALUE ? encode(embeddings.get(0), stories).add(temporal.get(0)) : null;
        for (int k = 0; k < hops; k++) {
            Var keys = kind == Kind.KEY_VALUE ? keyMemory
                    : encode(embeddings.get(k), stories).add(temporal.get(k));
            Var vals = kind == Kind.KEY_VALUE ? valueMemory
                    : encode(embeddings.get(k + 1), stories).add(temporal.get(k + 1));
            Var scores = keys.matmul(u.reshape(b, dim, 1)).reshape(b, sentences).add(maskBias);
            Var p = Activation.SOFTMAX.forward(scores);
            lastAttention[k] = p.value().slice(0, 0, 1).toArray();
            Var o = p.reshape(b, 1, sentences).matmul(vals).reshape(b, dim);
            u = kind == Kind.KEY_VALUE ? u.add(o).matmul(rotations.get(k)) : u.add(o);
        }
        return u.matmul(answer);
    }

    /**
     * The attention of the first example over its memories at each hop, from the last call
     * to {@link #logits}.
     *
     * @return {@code [hop][sentence]}
     */
    public float[][] lastAttention() {
        return lastAttention;
    }

    /** @return the number of hops */
    public int hops() {
        return hops;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(embeddings);
        p.addAll(temporal);
        if (values != null) {
            p.add(values);
        }
        p.addAll(rotations);
        p.add(answer);
        return p;
    }

    @Override
    public String toString() {
        return "MemoryNetwork(" + kind + ", " + hops + " hops, " + sentences + " slots)";
    }
}
