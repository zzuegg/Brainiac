package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Next-item recommendation from a user's item history with a transformer over item embeddings
 * and learned positions; the output layer is tied to the item embeddings.
 * <ul>
 *   <li>{@link #sasRec} (Kang and McAuley, 2018, <a href="https://arxiv.org/abs/1808.09781">
 *       arXiv:1808.09781</a>): causal core {@link TransformerBlock}s; every position predicts the
 *       next item;</li>
 *   <li>{@link #bert4Rec} (Sun et al., 2019, <a href="https://arxiv.org/abs/1904.06690">
 *       arXiv:1904.06690</a>): bidirectional blocks trained as a masked-item model (Cloze task);
 *       to recommend, a mask token is appended to the history and predicted;</li>
 *   <li>{@link #hstu} (Zhai et al., 2024, <a href="https://arxiv.org/abs/2402.17152">
 *       arXiv:2402.17152</a>): causal {@link HstuBlock}s with relative position and time
 *       biases, reading event timestamps; the stream also carries each event's action (click or
 *       skip), so one model serves retrieval and ranking.</li>
 * </ul>
 * All three can carry a click head that predicts whether the next item will be clicked, from the
 * hidden state and the candidate's embedding (ranking); events' actions are added to their item
 * embeddings when {@code actions} are given.
 */
public final class SequentialRecommender {

    private enum Kind {
        SAS_REC, BERT4REC, HSTU
    }

    private final Kind kind;
    private final int items;
    private final int length;
    private final int dim;
    private final Var embedding;
    private final Var positions;
    private final Var actionEmbedding;
    private final Var clickWeight;
    private final Var clickBias;
    private final Var gapEmbedding;
    private final List<MultiInputLayer> blocks = new ArrayList<>();
    private final SplittableRandom random;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private float maskRate = 0.2f;

    private SequentialRecommender(Kind kind, int items, int length, int dim, int heads, int layers, long seed) {
        this.kind = kind;
        this.items = items;
        this.length = length;
        this.dim = dim;
        this.random = new SplittableRandom(seed);
        embedding = RecsysOps.normal(random, 0.1f, items + 2, dim);
        positions = RecsysOps.normal(random, 0.1f, length, dim);
        actionEmbedding = RecsysOps.normal(random, 0.1f, 2, dim);
        clickWeight = RecsysOps.matrix(random, dim, 1);
        clickBias = RecsysOps.zeros(1);
        gapEmbedding = RecsysOps.normal(random, 0.1f, HstuBlock.TIME_BUCKETS, dim);
        Shape seq = Shape.of(length, dim);
        for (int l = 0; l < layers; l++) {
            blocks.add(switch (kind) {
                case SAS_REC -> (MultiInputLayer) TransformerBlock.of(heads).ffnSize(2 * dim).causal(true)
                        .create(seq, random);
                case BERT4REC -> (MultiInputLayer) TransformerBlock.of(heads).ffnSize(2 * dim).create(seq, random);
                case HSTU -> new HstuBlock(dim, heads, length, random);
            });
        }
    }

    /**
     * SASRec.
     *
     * @param items the number of items
     * @param length the history window
     * @param dim the model width
     * @param heads attention heads
     * @param layers transformer blocks
     * @param seed the seed
     * @return the model
     */
    public static SequentialRecommender sasRec(int items, int length, int dim, int heads, int layers, long seed) {
        return new SequentialRecommender(Kind.SAS_REC, items, length, dim, heads, layers, seed);
    }

    /**
     * BERT4Rec.
     *
     * @param items the number of items
     * @param length the history window, including the appended mask token
     * @param dim the model width
     * @param heads attention heads
     * @param layers transformer blocks
     * @param seed the seed
     * @return the model
     */
    public static SequentialRecommender bert4Rec(int items, int length, int dim, int heads, int layers, long seed) {
        return new SequentialRecommender(Kind.BERT4REC, items, length, dim, heads, layers, seed);
    }

    /**
     * A generative recommender of HSTU blocks.
     *
     * @param items the number of items
     * @param length the history window
     * @param dim the model width
     * @param heads attention heads
     * @param layers HSTU blocks
     * @param seed the seed
     * @return the model
     */
    public static SequentialRecommender hstu(int items, int length, int dim, int heads, int layers, long seed) {
        return new SequentialRecommender(Kind.HSTU, items, length, dim, heads, layers, seed);
    }

    /**
     * @param o the optimizer
     * @return this model
     */
    public SequentialRecommender optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /**
     * @param rate the share of positions BERT4Rec masks while training
     * @return this model
     */
    public SequentialRecommender maskRate(float rate) {
        this.maskRate = rate;
        return this;
    }

    /** @return the mask token id */
    public int maskToken() {
        return items + 1;
    }

    /** @return the history window */
    public int length() {
        return length;
    }

    /**
     * Hidden states of input windows.
     *
     * @param tokens {@code [batch][length]} item ids (or the mask token)
     * @param times {@code [batch][length]} timestamps, or {@code null}
     * @param actions {@code [batch][length]} 0/1 actions of the events, or {@code null}
     * @param training whether dropout-style layers train
     * @return {@code [batch, length, dim]}
     */
    public Var hidden(int[][] tokens, long[][] times, int[][] actions, boolean training) {
        int b = tokens.length;
        Var x = embedding.gather(0, RecsysOps.ids(tokens).reshape(b * length)).reshape(b, length, dim)
                .add(positions);
        if (actions != null) {
            x = x.add(actionEmbedding.gather(0, RecsysOps.ids(actions).reshape(b * length)).reshape(b, length, dim));
        }
        Var t = null;
        if (times != null && kind == Kind.HSTU) {
            float[] f = new float[b * length];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < length; j++) {
                    f[i * length + j] = times[i][j] - times[i][0];
                }
            }
            t = Var.of(Tensor.of(Shape.of(b, length), f));
        }
        for (MultiInputLayer block : blocks) {
            x = block.forward(t != null ? List.of(x, t) : List.of(x), training);
        }
        return x;
    }

    private Var itemLogits(Var h) {
        return h.matmul(embedding.slice(0, 0, items).transpose());
    }

    private static int gapBucket(long gap) {
        return Math.min(HstuBlock.TIME_BUCKETS - 1, (int) (Math.log(1 + Math.abs(gap)) / Math.log(2)));
    }

    private Var clickLogit(Var h, Tensor candidates, Tensor gaps) {
        return h.mul(embedding.gather(0, candidates).add(gapEmbedding.gather(0, gaps))).matmul(clickWeight)
                .add(clickBias);
    }

    /**
     * One training step on windows of {@code length + 1} consecutive events. SASRec and HSTU
     * predict event {@code t + 1} from events {@code ..t} at every position; BERT4Rec masks a
     * share of the last {@code length} events (the last one half of the time) and predicts them.
     * With {@code clicks}, the click head is also trained on whether each next event was clicked.
     *
     * @param windows {@code [batch][length + 1]} item ids
     * @param times {@code [batch][length + 1]} timestamps, or {@code null}
     * @param clicks {@code [batch][length + 1]} 0/1 actions, or {@code null}
     * @return the loss
     */
    public double trainStep(int[][] windows, long[][] times, int[][] clicks) {
        int b = windows.length;
        int[][] in = new int[b][length];
        long[][] tin = times == null ? null : new long[b][length];
        int[][] ain = clicks == null ? null : new int[b][length];
        float[] target = new float[b * length];
        float[] nextItem = new float[b * length];
        float[] nextClick = new float[b * length];
        float[] gaps = new float[b * length];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < length; j++) {
                if (kind == Kind.BERT4REC) {
                    int item = windows[i][j + 1];
                    boolean mask = j == length - 1 ? random.nextDouble() < 0.5 : random.nextDouble() < maskRate;
                    in[i][j] = mask ? maskToken() : item;
                    target[i * length + j] = mask ? item : -1;
                } else {
                    in[i][j] = windows[i][j];
                    target[i * length + j] = windows[i][j + 1];
                }
                nextItem[i * length + j] = windows[i][j + 1];
                if (tin != null) {
                    tin[i][j] = times[i][kind == Kind.BERT4REC ? j + 1 : j];
                }
                if (ain != null) {
                    ain[i][j] = kind == Kind.BERT4REC ? 0 : clicks[i][j];
                    nextClick[i * length + j] = clicks[i][j + 1];
                    gaps[i * length + j] = times == null ? 0 : gapBucket(times[i][j + 1] - times[i][j]);
                }
            }
        }
        Var h = hidden(in, tin, ain, true).reshape(b * length, dim);
        Var loss = Functional.crossEntropy(itemLogits(h), Tensor.of(Shape.of(b * length), target));
        if (clicks != null && kind != Kind.BERT4REC) {
            Var z = clickLogit(h, Tensor.of(Shape.of(b * length), nextItem), Tensor.of(Shape.of(b * length), gaps));
            loss = loss.add(z.softplus().sub(z.mul(Tensor.of(Shape.of(b * length, 1), nextClick))).mean());
        }
        loss.backward();
        optimizer.step(parameters());
        return loss.value().item();
    }

    private int[] window(int[] history) {
        int[] w = new int[length];
        if (kind == Kind.BERT4REC) {
            System.arraycopy(history, history.length - (length - 1), w, 0, length - 1);
            w[length - 1] = maskToken();
        } else {
            System.arraycopy(history, history.length - length, w, 0, length);
        }
        return w;
    }

    private long[] window(long[] times) {
        long[] w = new long[length];
        System.arraycopy(times, times.length - length, w, 0, length);
        return w;
    }

    private int[] actionWindow(int[] actions) {
        int[] w = new int[length];
        if (kind != Kind.BERT4REC) {
            System.arraycopy(actions, actions.length - length, w, 0, length);
        }
        return w;
    }

    /**
     * Next-item scores for a batch of histories.
     *
     * @param histories {@code [batch][>= length]} item ids, oldest first
     * @param times their timestamps, or {@code null}
     * @param actions their 0/1 actions, or {@code null}
     * @return {@code [batch][items]} logits
     */
    public float[][] scores(int[][] histories, long[][] times, int[][] actions) {
        int b = histories.length;
        int[][] in = new int[b][];
        long[][] tin = times == null ? null : new long[b][];
        int[][] ain = actions == null ? null : new int[b][];
        for (int i = 0; i < b; i++) {
            in[i] = window(histories[i]);
            if (tin != null) {
                tin[i] = window(times[i]);
            }
            if (ain != null) {
                ain[i] = actionWindow(actions[i]);
            }
        }
        return Var.noGrad(() -> {
            Var h = hidden(in, tin, ain, false).slice(1, length - 1, length).reshape(b, dim);
            return RecsysOps.rows(itemLogits(h).value());
        });
    }

    /**
     * Click logits of candidate next items (the ranking head).
     *
     * @param histories {@code [batch][>= length]} item ids, oldest first
     * @param times their timestamps, or {@code null}
     * @param actions their 0/1 actions, or {@code null}
     * @param candidates one candidate next item per history
     * @param candidateTimes the request time of each candidate (the click head reads the gap
     *     since the last event), or {@code null}; needs {@code times}
     * @return one click logit per history
     */
    public float[] clickLogits(int[][] histories, long[][] times, int[][] actions, int[] candidates,
            long[] candidateTimes) {
        float[] gaps = new float[candidates.length];
        for (int i = 0; candidateTimes != null && i < gaps.length; i++) {
            gaps[i] = gapBucket(candidateTimes[i] - times[i][times[i].length - 1]);
        }
        int b = histories.length;
        int[][] in = new int[b][];
        long[][] tin = times == null ? null : new long[b][];
        int[][] ain = actions == null ? null : new int[b][];
        for (int i = 0; i < b; i++) {
            in[i] = window(histories[i]);
            if (tin != null) {
                tin[i] = window(times[i]);
            }
            if (ain != null) {
                ain[i] = actionWindow(actions[i]);
            }
        }
        return Var.noGrad(() -> {
            Var h = hidden(in, tin, ain, false).slice(1, length - 1, length).reshape(b, dim);
            return clickLogit(h, RecsysOps.ids(candidates), Tensor.of(Shape.of(gaps.length), gaps)).value()
                    .toArray();
        });
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding, positions, actionEmbedding, clickWeight, clickBias,
                gapEmbedding));
        for (Layer block : blocks) {
            p.addAll(block.parameters());
        }
        return p;
    }

    /** @return the number of trainable scalars */
    public long size() {
        long n = 0;
        for (Var v : parameters()) {
            n += v.shape().size();
        }
        return n;
    }

    @Override
    public String toString() {
        return kind + "(" + items + " items, width " + dim + ", " + blocks.size() + " blocks)";
    }
}
