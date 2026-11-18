package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ResidualVectorQuantizer;
import io.github.zzuegg.brainiac.experimental.nn.VectorQuantizer;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Semantic IDs for items, as in TIGER (Rajput et al., 2023, <a href="https://arxiv.org/abs/2305.05065">
 * arXiv:2305.05065</a>): each item's content embedding is quantized by a residual vector
 * quantizer ({@link ResidualVectorQuantizer}) into a short tuple of codewords, coarse to fine, so
 * items with similar content share ID prefixes. Items whose codes collide get one more token, their
 * index among the colliding items, so every item has a unique ID of {@code depth + 1} tokens. New
 * items get an ID from their content alone (cold start).
 *
 * <p>The paper trains an RQ-VAE (an encoder and decoder around the quantizer); this block quantizes
 * the content embeddings directly: each stage is seeded with k-means++ on its residuals and then
 * refined with the quantizer's codebook loss.
 */
public final class SemanticIdTokenizer {

    private final int depth;
    private final int codes;
    private final int[][] ids;
    private final int tokensPerLevel;
    private final double collisionRate;

    private SemanticIdTokenizer(int depth, int codes, int[][] ids, int tokensPerLevel, double collisionRate) {
        this.depth = depth;
        this.codes = codes;
        this.ids = ids;
        this.tokensPerLevel = tokensPerLevel;
        this.collisionRate = collisionRate;
    }

    /**
     * Fits the quantizer and assigns every item its ID.
     *
     * @param content {@code [items, dim]} content embeddings
     * @param depth codebooks (levels before the disambiguation token)
     * @param codes codes per codebook
     * @param steps refinement steps of the codebooks
     * @param seed the seed
     * @return the tokenizer
     */
    public static SemanticIdTokenizer fit(Tensor content, int depth, int codes, int steps, long seed) {
        int dim = content.shape().dim(1);
        ResidualVectorQuantizer rvq = ResidualVectorQuantizer.of(depth, codes, dim, 0.25f, seed);
        Tensor residual = content;
        for (VectorQuantizer q : rvq.stages()) {
            q.initialize(residual);
            residual = residual.sub(q.lookup(q.nearest(residual)).value());
        }
        Optimizer opt = Optimizer.adam(0.01f);
        List<Var> params = rvq.parameters();
        for (int s = 0; s < steps; s++) {
            rvq.quantize(Var.of(content), depth).loss().backward();
            opt.step(params);
        }
        int[][] stageCodes = rvq.quantize(Var.of(content), depth).codes();
        int items = content.shape().dim(0);
        int[][] ids = new int[items][depth + 1];
        Map<String, Integer> seen = new HashMap<>();
        int collided = 0;
        int maxIndex = 0;
        for (int i = 0; i < items; i++) {
            StringBuilder key = new StringBuilder();
            for (int l = 0; l < depth; l++) {
                ids[i][l] = stageCodes[l][i];
                key.append(ids[i][l]).append(',');
            }
            int index = seen.merge(key.toString(), 1, Integer::sum) - 1;
            ids[i][depth] = index;
            collided += index > 0 ? 1 : 0;
            maxIndex = Math.max(maxIndex, index);
        }
        return new SemanticIdTokenizer(depth, codes, ids, Math.max(codes, maxIndex + 1), collided / (double) items);
    }

    /** @return the number of tokens per ID, {@code depth + 1} */
    public int length() {
        return depth + 1;
    }

    /** @return the codes per codebook */
    public int codes() {
        return codes;
    }

    /** @return the number of distinct tokens per level (codes, or more for many collisions) */
    public int tokensPerLevel() {
        return tokensPerLevel;
    }

    /** @return the share of items whose codes collided with an earlier item's */
    public double collisionRate() {
        return collisionRate;
    }

    /** @return the number of items */
    public int items() {
        return ids.length;
    }

    /**
     * @param item an item
     * @return its ID, {@code depth + 1} codes
     */
    public int[] id(int item) {
        return ids[item].clone();
    }

    /**
     * The ID as tokens of one shared vocabulary: level {@code l}'s code {@code c} is token
     * {@code l * tokensPerLevel() + c}.
     *
     * @param item an item
     * @return {@code depth + 1} token ids
     */
    public int[] tokens(int item) {
        int[] t = new int[depth + 1];
        for (int l = 0; l <= depth; l++) {
            t[l] = l * tokensPerLevel + ids[item][l];
        }
        return t;
    }

    /** @return the size of the shared token vocabulary */
    public int vocabulary() {
        return (depth + 1) * tokensPerLevel;
    }
}
