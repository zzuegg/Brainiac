package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learned sparse retrieval, SPLADE (Formal et al., 2021, <a href="https://arxiv.org/abs/2107.05720">
 * arXiv:2107.05720</a>). A masked-language-model head maps every token's hidden state to logits
 * over the whole vocabulary (tied to the input embeddings); a text's representation is
 * {@code w_v = max_t log(1 + ReLU(logit_tv))}, a bag of weighted vocabulary terms that includes
 * <em>expansion</em> terms absent from the text (such as synonyms). Relevance is the dot product of
 * two such bags. Training is contrastive (InfoNCE with in-batch negatives) plus the FLOPS
 * regulariser {@code sum_v (mean_batch w_v)^2}, which pushes towards sparse bags that an inverted
 * index ({@link Bm25Index#impact}) serves cheaply.
 */
public final class SparseLexicalEncoder {

    private final TokenEncoder encoder;
    private final Var bias;
    private final float flopsWeight;
    private Optimizer optimizer = Optimizer.adam(3e-3f);

    /**
     * Creates the encoder.
     *
     * @param vocabulary the vocabulary size, including padding token 0
     * @param maxLength the longest query or document
     * @param dim the width
     * @param heads attention heads
     * @param layers transformer blocks
     * @param flopsWeight the weight of the FLOPS sparsity regulariser
     * @param seed the seed
     */
    public SparseLexicalEncoder(int vocabulary, int maxLength, int dim, int heads, int layers, float flopsWeight,
            long seed) {
        this.encoder = new TokenEncoder(vocabulary, maxLength, dim, heads, layers, new SplittableRandom(seed));
        this.bias = RecsysOps.zeros(vocabulary);
        this.flopsWeight = flopsWeight;
    }

    /**
     * @param o the optimizer
     * @return this encoder
     */
    public SparseLexicalEncoder optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /**
     * The sparse term weights of texts.
     *
     * @param tokens {@code [batch][length]}
     * @param training whether dropout-style layers train
     * @return {@code [batch, vocabulary]}, non-negative
     */
    public Var weights(int[][] tokens, boolean training) {
        int b = tokens.length;
        int len = tokens[0].length;
        int v = encoder.vocabulary();
        Var logits = encoder.encode(tokens, training).reshape(b * len, encoder.dim())
                .matmul(encoder.embeddings().transpose()).add(bias);
        Var w = logits.relu().add(1).log().reshape(b, len, v).mul(TokenEncoder.mask(tokens));
        return w.max(1, false);
    }

    private static Var flops(Var w) {
        return w.mean(0, false).square().sum();
    }

    /**
     * One contrastive step with the FLOPS regulariser on both sides.
     *
     * @param queries {@code [batch][length]}
     * @param positives {@code [batch][length]}
     * @param hardNegatives {@code [batch][length]}, or {@code null}
     * @return the loss
     */
    public double trainStep(int[][] queries, int[][] positives, int[][] hardNegatives) {
        Var q = weights(queries, true);
        Var d = weights(positives, true);
        if (hardNegatives != null) {
            d = Var.concat(0, d, weights(hardNegatives, true));
        }
        Var loss = Functional.crossEntropy(q.matmul(d.transpose()), Tensor.arange(queries.length))
                .add(flops(q).add(flops(d)).mul(flopsWeight));
        loss.backward();
        List<Var> p = new ArrayList<>(encoder.parameters());
        p.add(bias);
        optimizer.step(p);
        return loss.value().item();
    }

    /**
     * Encodes documents into an impact index, keeping only their non-zero terms.
     *
     * @param documents {@code [n][length]}
     * @return the index
     */
    public Bm25Index index(int[][] documents) {
        Bm25Index index = Bm25Index.impact();
        int v = encoder.vocabulary();
        float[] w = Var.noGrad(() -> weights(documents, false)).value().toArray();
        for (int i = 0; i < documents.length; i++) {
            List<Integer> terms = new ArrayList<>();
            List<Float> vals = new ArrayList<>();
            for (int t = 1; t < v; t++) {
                float x = w[i * v + t];
                if (x > 0) {
                    terms.add(t);
                    vals.add(x);
                }
            }
            float[] f = new float[vals.size()];
            for (int k = 0; k < f.length; k++) {
                f[k] = vals.get(k);
            }
            index.add(terms.stream().mapToInt(Integer::intValue).toArray(), f);
        }
        return index;
    }

    /**
     * Scores every indexed document for a query.
     *
     * @param index an index from {@link #index}
     * @param query the query's tokens
     * @return one score per document
     */
    public float[] scores(Bm25Index index, int[] query) {
        int v = encoder.vocabulary();
        float[] w = Var.noGrad(() -> weights(new int[][] {query}, false)).value().toArray();
        List<Integer> terms = new ArrayList<>();
        List<Float> vals = new ArrayList<>();
        for (int t = 1; t < v; t++) {
            if (w[t] > 0) {
                terms.add(t);
                vals.add(w[t]);
            }
        }
        float[] f = new float[vals.size()];
        for (int k = 0; k < f.length; k++) {
            f[k] = vals.get(k);
        }
        return index.scores(terms.stream().mapToInt(Integer::intValue).toArray(), f);
    }
}
