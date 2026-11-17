package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * The Free Transformer: a decoder-only language model that conditions its second half on
 * random latent variables {@code Z}, learned without supervision as a conditional VAE. The
 * model can "decide" global properties of a sequence (a style, a plan, a direction) by
 * drawing {@code Z} before writing, instead of inferring them token by token from what it
 * has already sampled.
 *
 * <ul>
 *   <li>Decoder: embedding, the first half of the blocks, then {@code h += Z W_z}, then the
 *       second half and the unembedding.</li>
 *   <li>Encoder (training only): one non-causal block over the middle activations (plus a
 *       learned query embedding) predicts {@code H} bit logits per token; the bits are
 *       sampled and turned into a one-hot {@code Z} over {@code 2^H} codes with a
 *       straight-through gradient.</li>
 *   <li>Loss: next-token cross-entropy plus {@code max(0, KL(Q(Z|x) || uniform) - freeBits)},
 *       so the latent only carries information worth its cost.</li>
 *   <li>Generation: {@code Z} is drawn uniformly, or fixed by the caller.</li>
 * </ul>
 *
 * <p>Paper: Fleuret, "The Free Transformer", <a href="https://arxiv.org/abs/2510.17558">arXiv:2510.17558</a>.
 */
public final class FreeTransformer {

    /**
     * The terms of the training loss.
     *
     * @param total the loss to minimize
     * @param reconstruction the next-token cross-entropy
     * @param kl the mean KL divergence per token, in nats
     */
    public record Losses(Var total, Var reconstruction, Var kl) {
    }

    /** A pre-norm block: {@code x + attention(norm x) + mlp(norm ...)}. */
    private static final class Block {
        final GatedAttention attention;
        final Var w1;
        final Var w2;

        Block(int d, boolean causal, RandomGenerator random) {
            attention = new GatedAttention(d, 2, causal, 10000f, random);
            w1 = MixerOps.weight(random, d, 2 * d);
            w2 = MixerOps.weight(random, 2 * d, d);
        }

        Var forward(Var x) {
            x = x.add(attention.forward(MixerOps.rmsNorm(x)));
            return x.add(MixerOps.mlp(MixerOps.rmsNorm(x), w1, w2));
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(attention.parameters());
            p.add(w1);
            p.add(w2);
            return p;
        }
    }

    private final int vocabulary;
    private final int features;
    private final int bits;
    private final int codes;
    private final float freeBits;
    private final Var embedding;
    private final Var unembedding;
    private final List<Block> lower = new ArrayList<>();
    private final List<Block> upper = new ArrayList<>();
    private final Block encoder;
    private final Var encoderQuery;
    private final Var toBits;
    private final Var fromLatent;
    private final Tensor bitTable;

    /**
     * Creates the model.
     *
     * @param vocabulary the number of token ids
     * @param features the model width, even
     * @param blocks the number of decoder blocks, at least 2 (the latent enters in the middle)
     * @param bits {@code H}: the latent is one of {@code 2^H} codes per token
     * @param freeBits the KL allowance per token in nats, below which the KL is not penalized
     * @param random the random source
     */
    public FreeTransformer(int vocabulary, int features, int blocks, int bits, float freeBits, RandomGenerator random) {
        if (blocks < 2 || bits < 1 || bits > 12) {
            throw new IllegalArgumentException("need at least 2 blocks and 1..12 bits");
        }
        this.vocabulary = vocabulary;
        this.features = features;
        this.bits = bits;
        this.codes = 1 << bits;
        this.freeBits = freeBits;
        this.embedding = Var.param(Tensor.normal(Shape.of(vocabulary, features), 0, 1, random));
        this.unembedding = MixerOps.weight(random, features, vocabulary);
        for (int i = 0; i < blocks; i++) {
            (i < blocks / 2 ? lower : upper).add(new Block(features, true, random));
        }
        this.encoder = new Block(features, false, random);
        this.encoderQuery = MixerOps.uniform(random, 0.1f, features);
        this.toBits = MixerOps.weight(random, features, bits);
        this.fromLatent = MixerOps.weight(random, codes, features);
        float[] table = new float[bits * codes];
        for (int c = 0; c < codes; c++) {
            for (int h = 0; h < bits; h++) {
                table[h * codes + c] = (c >> h) & 1;
            }
        }
        this.bitTable = Tensor.of(Shape.of(bits, codes), table);
    }

    /** @return {@code 2^H}, the number of latent codes */
    public int codes() {
        return codes;
    }

    private Var embed(int[][] tokens) {
        int b = tokens.length;
        int t = tokens[0].length;
        float[] ids = new float[b * t];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                ids[i * t + j] = tokens[i][j];
            }
        }
        return embedding.gather(0, Tensor.of(Shape.of(b * t), ids)).reshape(b, t, features);
    }

    private Var lowerHalf(int[][] tokens) {
        Var h = embed(tokens);
        for (Block blk : lower) {
            h = blk.forward(h);
        }
        return h;
    }

    private Var upperHalf(Var h, Var z) {
        h = h.add(z.matmul(fromLatent));
        for (Block blk : upper) {
            h = blk.forward(h);
        }
        return MixerOps.rmsNorm(h).matmul(unembedding);
    }

    /**
     * Computes the VAE training loss on a batch.
     *
     * @param tokens {@code [batch][steps]}
     * @param random the source for sampling the latent bits
     * @return the loss terms
     */
    public Losses loss(int[][] tokens, RandomGenerator random) {
        int b = tokens.length;
        int t = tokens[0].length;
        Var mid = lowerHalf(tokens);
        Var logits = encoder.forward(mid.add(encoderQuery)).matmul(toBits);   // [b, t, H]
        Var logP = logits.neg().softplus().neg();   // log sigmoid
        Var logQ = logits.softplus().neg();          // log (1 - sigmoid)
        Var bitsTable = Var.of(bitTable);
        Var logCode = logP.matmul(bitsTable).add(logQ.matmul(Var.of(bitTable.neg().add(1f))));   // [b, t, codes]
        Var p = logCode.exp();
        // sample each bit, then the straight-through one-hot
        float[] prob = logits.sigmoid().value().toArray();
        float[] onehot = new float[b * t * codes];
        for (int i = 0; i < b * t; i++) {
            int c = 0;
            for (int h = 0; h < bits; h++) {
                if (random.nextFloat() < prob[i * bits + h]) {
                    c |= 1 << h;
                }
            }
            onehot[i * codes + c] = 1;
        }
        Var z = Functional.straightThrough(Var.of(Tensor.of(Shape.of(b, t, codes), onehot)), p);
        Var out = upperHalf(mid, z);
        float[] target = new float[b * t];
        for (int n = 0; n < b; n++) {
            for (int s = 0; s < t; s++) {
                target[n * t + s] = s + 1 < t ? tokens[n][s + 1] : -1;
            }
        }
        Var rec = Functional.crossEntropy(out.reshape(b * t, vocabulary), Tensor.of(Shape.of(b * t), target));
        Var kl = p.mul(logCode.add((float) Math.log(codes))).sum(-1, false);   // [b, t]
        Var klMean = kl.mean();
        Var penalty = kl.sub(freeBits).relu().mean();
        return new Losses(rec.add(penalty), rec, klMean);
    }

    /**
     * Generates greedily with a given latent code at every step.
     *
     * @param prompt the prompt, at least one token
     * @param count how many tokens to generate
     * @param code the latent code in {@code [0, codes)} used at every position
     * @return the generated tokens
     */
    public int[] generate(int[] prompt, int count, int code) {
        return generate(prompt, count, i -> code);
    }

    /**
     * Generates greedily with a uniformly random latent code per position, as at inference.
     *
     * @param prompt the prompt, at least one token
     * @param count how many tokens to generate
     * @param random the random source for the latent
     * @return the generated tokens
     */
    public int[] generate(int[] prompt, int count, RandomGenerator random) {
        int[] fixed = new int[prompt.length + count];
        for (int i = 0; i < fixed.length; i++) {
            fixed[i] = random.nextInt(codes);
        }
        return generate(prompt, count, i -> fixed[i]);
    }

    private int[] generate(int[] prompt, int count, java.util.function.IntUnaryOperator codeAt) {
        int[] seq = prompt.clone();
        for (int k = 0; k < count; k++) {
            int t = seq.length;
            float[] onehot = new float[t * codes];
            for (int i = 0; i < t; i++) {
                onehot[i * codes + codeAt.applyAsInt(i)] = 1;
            }
            Var out = upperHalf(lowerHalf(new int[][] {seq}), Var.of(Tensor.of(Shape.of(1, t, codes), onehot)));
            float[] row = out.value().slice(1, t - 1, t).toArray();
            int best = 0;
            for (int i = 1; i < row.length; i++) {
                if (row[i] > row[best]) {
                    best = i;
                }
            }
            seq = Arrays.copyOf(seq, t + 1);
            seq[t] = best;
        }
        return Arrays.copyOfRange(seq, prompt.length, seq.length);
    }

    /** @return all trainable parameters, encoder included */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding, unembedding, encoderQuery, toBits, fromLatent));
        lower.forEach(b -> p.addAll(b.parameters()));
        upper.forEach(b -> p.addAll(b.parameters()));
        p.addAll(encoder.parameters());
        return p;
    }
}
