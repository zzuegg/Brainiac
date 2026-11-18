package io.github.zzuegg.brainiac.experimental.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The BLIP-2 bridge, a Q-Former (Li et al., ICML 2023): a small transformer with learned
 * query tokens that attend among themselves and, by cross-attention, to a frozen vision
 * encoder's patch features (core {@link TransformerBlock}s with cross-attention). Its few
 * output queries are projected into the language model's embedding space as a short visual
 * prefix, so the language model sees a handful of text-relevant visual tokens instead of
 * every patch.
 *
 * <p>BLIP-2 first pretrains the Q-Former against text without the language model;
 * {@link #contrastiveLoss} is its image-text contrastive objective (each image's best
 * query against the caption embedding, symmetric InfoNCE). The matching and captioning
 * objectives of the paper are not included; the captioning signal comes from training the
 * prefix through the frozen language model afterwards.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2301.12597">arXiv:2301.12597</a>.
 */
public final class QueryTransformer {

    private final int dim;
    private final Var inProject;
    private final Var queries;
    private final List<MultiInputLayer> blocks = new ArrayList<>();
    private final Var outProject;
    private final Var textProject;

    /**
     * Creates the Q-Former.
     *
     * @param visionDim the patch feature width
     * @param dim the Q-Former width
     * @param lmDim the language model width
     * @param textDim the width of the text embeddings used by {@link #contrastiveLoss}
     * @param queries the number of learned queries
     * @param layers blocks
     * @param heads attention heads
     * @param seed the seed
     */
    public QueryTransformer(int visionDim, int dim, int lmDim, int textDim, int queries, int layers, int heads,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.dim = dim;
        this.inProject = Var.param(Tensor.normal(Shape.of(visionDim, dim), 0, (float) Math.sqrt(1.0 / visionDim), r));
        this.queries = Var.param(Tensor.normal(Shape.of(1, queries, dim), 0, 0.3f, r));
        for (int i = 0; i < layers; i++) {
            blocks.add(TransformerBlock.of(heads).ffnSize(2 * dim).crossAttention(true)
                    .create(List.of(Shape.of(8, dim), Shape.of(8, dim)), r));
        }
        this.outProject = Var.param(Tensor.normal(Shape.of(dim, lmDim), 0, (float) Math.sqrt(1.0 / dim), r));
        this.textProject = Var.param(Tensor.normal(Shape.of(textDim, dim), 0, (float) Math.sqrt(1.0 / textDim), r));
    }

    /**
     * The query outputs in the Q-Former's own width.
     *
     * @param patches {@code [batch, patches, visionDim]}
     * @return {@code [batch, queries, dim]}
     */
    public Var queries(Var patches) {
        int b = patches.shape().dim(0);
        Var memory = patches.matmul(inProject);
        Var x = queries.expand(Shape.of(b, queries.shape().dim(1), dim));
        for (MultiInputLayer l : blocks) {
            x = l.forward(List.of(x, memory), false);
        }
        return x;
    }

    /**
     * The visual prefix for the language model.
     *
     * @param patches {@code [batch, patches, visionDim]}
     * @return {@code [batch, queries, lmDim]}
     */
    public Var prefix(Var patches) {
        return queries(patches).matmul(outProject);
    }

    /**
     * Image-text contrastive loss: similarity of an image and a text is the best cosine
     * similarity of any query with the projected text embedding; symmetric InfoNCE over the
     * batch, matching pairs on the diagonal.
     *
     * @param patches {@code [batch, patches, visionDim]}
     * @param text {@code [batch, textDim]} text embeddings (for example a frozen LM's mean hidden state)
     * @param temperature the softmax temperature, such as 0.1
     * @return the loss
     */
    public Var contrastiveLoss(Var patches, Var text, float temperature) {
        int b = patches.shape().dim(0);
        int nq = queries.shape().dim(1);
        Var q = unit(queries(patches));                    // [b, nq, dim]
        Var t = unit(text.matmul(textProject));            // [b, dim]
        // sim[i, j, k] = q[i, k] . t[j]
        Var sims = q.reshape(b * nq, dim).matmul(t.transpose()).reshape(b, nq, b).permute(0, 2, 1); // [b, b, nq]
        Var logits = sims.max(2, false).mul(1f / temperature);           // [b images, b texts]
        float[] diag = new float[b];
        for (int i = 0; i < b; i++) {
            diag[i] = i;
        }
        Tensor ids = Tensor.of(Shape.of(b), diag);
        return Functional.crossEntropy(logits, ids).add(Functional.crossEntropy(logits.transpose(), ids)).mul(0.5f);
    }

    private static Var unit(Var x) {
        int last = x.shape().rank() - 1;
        return x.div(x.square().sum(last, true).add(1e-6f).sqrt());
    }

    /** @return every trainable value */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(inProject, queries));
        for (MultiInputLayer l : blocks) {
            p.addAll(l.parameters());
        }
        p.add(outProject);
        p.add(textProject);
        return p;
    }

    @Override
    public String toString() {
        return "QueryTransformer(" + queries.shape().dim(1) + " queries, " + blocks.size() + " blocks)";
    }
}
