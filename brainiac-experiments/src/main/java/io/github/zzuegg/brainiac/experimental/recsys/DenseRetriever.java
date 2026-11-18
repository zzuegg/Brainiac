package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.VectorIndex;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Dense passage retrieval, DPR (Karpukhin et al., 2020, <a href="https://arxiv.org/abs/2004.04906">
 * arXiv:2004.04906</a>): a dual encoder maps queries and documents to unit vectors
 * ({@link TokenEncoder} plus mean pooling) and relevance is their dot product, so documents are
 * encoded once, offline, into a core {@link VectorIndex}. Training is InfoNCE over in-batch
 * negatives (every other query's positive) plus optional hard negatives, such as BM25's top
 * non-relevant hit for each query, which teach the encoder what lexical overlap alone gets wrong.
 * Dense matching captures paraphrases that share no term with the document; it is weaker on rare
 * exact terms (names, ids) that the training data never taught it.
 */
public final class DenseRetriever {

    private final TokenEncoder queryEncoder;
    private final TokenEncoder docEncoder;
    private float temperature = 0.1f;
    private Optimizer optimizer = Optimizer.adam(3e-3f);

    /**
     * Creates the retriever.
     *
     * @param vocabulary the vocabulary size, including padding token 0
     * @param maxLength the longest query or document
     * @param dim the embedding width
     * @param heads attention heads
     * @param layers transformer blocks per encoder
     * @param shared whether queries and documents share one encoder (siamese) or use two (DPR)
     * @param seed the seed
     */
    public DenseRetriever(int vocabulary, int maxLength, int dim, int heads, int layers, boolean shared, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        queryEncoder = new TokenEncoder(vocabulary, maxLength, dim, heads, layers, r);
        docEncoder = shared ? queryEncoder : new TokenEncoder(vocabulary, maxLength, dim, heads, layers, r);
    }

    /**
     * @param t the softmax temperature of the contrastive loss
     * @return this retriever
     */
    public DenseRetriever temperature(float t) {
        this.temperature = t;
        return this;
    }

    /**
     * @param o the optimizer
     * @return this retriever
     */
    public DenseRetriever optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /**
     * @param query whether to use the query encoder
     * @return that encoder
     */
    public TokenEncoder encoder(boolean query) {
        return query ? queryEncoder : docEncoder;
    }

    /**
     * Embeds texts.
     *
     * @param tokens {@code [batch][length]} token ids
     * @param query whether these are queries
     * @param training whether dropout-style layers train
     * @return {@code [batch, dim]} unit vectors
     */
    public Var embed(int[][] tokens, boolean query, boolean training) {
        TokenEncoder e = encoder(query);
        return RecsysOps.normalize(TokenEncoder.meanPool(e.encode(tokens, training), tokens));
    }

    /**
     * The contrastive logits of queries against candidates.
     *
     * @param queries {@code [batch, dim]}
     * @param candidates {@code [m, dim]}
     * @return {@code [batch, m]}
     */
    public Var logits(Var queries, Var candidates) {
        return queries.matmul(candidates.transpose()).div(temperature);
    }

    /**
     * One InfoNCE step: query {@code i}'s positive is document {@code i}; all other documents of
     * the batch, and the hard negatives, are negatives.
     *
     * @param queries {@code [batch][length]}
     * @param positives {@code [batch][length]}
     * @param hardNegatives {@code [batch][length]}, or {@code null}
     * @return the loss
     */
    public double trainStep(int[][] queries, int[][] positives, int[][] hardNegatives) {
        Var q = embed(queries, true, true);
        Var d = embed(positives, false, true);
        if (hardNegatives != null) {
            d = Var.concat(0, d, embed(hardNegatives, false, true));
        }
        Var loss = Functional.crossEntropy(logits(q, d), Tensor.arange(queries.length));
        loss.backward();
        optimizer.step(parameters());
        return loss.value().item();
    }

    /**
     * Encodes documents into an exact inner-product index.
     *
     * @param documents {@code [n][length]}
     * @return the index, document {@code i} under id {@code i}
     */
    public VectorIndex index(int[][] documents) {
        Tensor e = Var.noGrad(() -> embed(documents, false, false)).value();
        VectorIndex index = VectorIndex.flat(e.shape().dim(1), VectorIndex.Metric.DOT, Device.cpu());
        index.add(e);
        return index;
    }

    /**
     * Scores every indexed document for a query.
     *
     * @param index an index from {@link #index}
     * @param query the query's tokens
     * @return one score per document
     */
    public float[] scores(VectorIndex index, int[] query) {
        Tensor q = Var.noGrad(() -> embed(new int[][] {query}, true, false)).value();
        return index.scores(q.reshape(q.shape().dim(1)));
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(queryEncoder.parameters());
        if (docEncoder != queryEncoder) {
            p.addAll(docEncoder.parameters());
        }
        return p;
    }

    /**
     * Rows of a padded token matrix, for building batches.
     *
     * @param rows token rows
     * @param length the padded length
     * @return {@code [rows][length]}, padded with 0
     */
    public static int[][] pad(List<int[]> rows, int length) {
        int[][] out = new int[rows.size()][length];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(rows.get(i), 0, out[i], 0, Math.min(length, rows.get(i).length));
        }
        return out;
    }
}
