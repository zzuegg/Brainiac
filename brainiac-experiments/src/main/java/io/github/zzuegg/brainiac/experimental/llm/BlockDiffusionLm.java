package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Block diffusion (BD3-LM, Arriola et al. 2025, "Block Diffusion: Interpolating Between
 * Autoregressive and Diffusion Language Models", <a href="https://arxiv.org/abs/2503.09573">arXiv:2503.09573</a>):
 * autoregressive over blocks of {@code B} tokens and masked diffusion within each block.
 * Block size 1 is an autoregressive model; block size equal to the length is a masked
 * diffusion model like {@link MaskedDiffusion}. In between it keeps what each side is good
 * at: a KV cache over finished blocks, generation of any length (stop after the block that
 * holds the end token) and several tokens per forward pass.
 *
 * <p>Training uses the paper's vectorized form: the transformer reads the noisy sequence
 * and the clean sequence side by side ({@code 2L} tokens) under a block-causal mask, noisy
 * block {@code b} seeing itself and clean blocks before {@code b}, clean block {@code b}
 * seeing clean blocks up to {@code b}, so every block's denoising loss is computed in one
 * pass. Each block draws its own mask ratio {@code t = k / B}; the loss is the masked
 * cross-entropy weighted by {@code 1 / t}, an upper bound on the negative log-likelihood
 * (NELBO, exact for {@code B = 1}). The transformer is built here from core layers and
 * {@link Functional#attention} with an explicit bias, since core transformer blocks only
 * take causal or full masks.
 */
public final class BlockDiffusionLm {

    private final int vocabulary;
    private final int maskId;
    private final int eosId;
    private final int length;
    private final int block;
    private final int width;
    private final int heads;
    private final Var embedding;
    private final Var positions;
    private final List<Layer> layers = new ArrayList<>();
    private final LayerNorm finalNorm;
    private final Dense head;
    private final Tensor trainingBias;

    private record Layer(LayerNorm norm1, Dense q, Dense k, Dense v, Dense o, LayerNorm norm2, Dense up,
            Dense down) {
        List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (var l : List.of(norm1, q, k, v, o, norm2, up, down)) {
                p.addAll(l.parameters());
            }
            return p;
        }
    }

    private BlockDiffusionLm(int vocabulary, int maskId, int eosId, int length, int block, int width, int layers,
            int heads, long seed) {
        if (length % block != 0 || width % heads != 0) {
            throw new IllegalArgumentException("length must be a multiple of the block, width of the heads");
        }
        this.vocabulary = vocabulary;
        this.maskId = maskId;
        this.eosId = eosId;
        this.length = length;
        this.block = block;
        this.width = width;
        this.heads = heads;
        SplittableRandom r = new SplittableRandom(seed);
        this.embedding = Var.param(Tensor.normal(Shape.of(vocabulary, width), 0, 0.1f, r));
        this.positions = Var.param(Tensor.normal(Shape.of(length, width), 0, 0.1f, r));
        for (int l = 0; l < layers; l++) {
            this.layers.add(new Layer(LayerNorm.of(width), dense(width, width, r), dense(width, width, r),
                    dense(width, width, r), dense(width, width, r), LayerNorm.of(width),
                    new Dense(width, 4 * width, Activation.GELU, Init.HE, r), dense(4 * width, width, r)));
        }
        this.finalNorm = LayerNorm.of(width);
        this.head = dense(width, vocabulary, r);
        this.trainingBias = trainingMask();
    }

    private static Dense dense(int in, int out, RandomGenerator r) {
        return new Dense(in, out, Activation.IDENTITY, Init.XAVIER, r);
    }

    /**
     * @param vocabulary the vocabulary, including the mask and end tokens
     * @param maskId the mask token
     * @param eosId the end token; sequences shorter than {@code length} are padded with it
     * @param length the longest sequence, a multiple of {@code block}
     * @param block the block size {@code B}
     * @param width the transformer width
     * @param layers transformer layers
     * @param heads attention heads
     * @param seed the seed
     * @return the model
     */
    public static BlockDiffusionLm of(int vocabulary, int maskId, int eosId, int length, int block, int width,
            int layers, int heads, long seed) {
        return new BlockDiffusionLm(vocabulary, maskId, eosId, length, block, width, layers, heads, seed);
    }

    /** @return the block size */
    public int blockSize() {
        return block;
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding, positions));
        for (Layer l : layers) {
            p.addAll(l.parameters());
        }
        p.addAll(finalNorm.parameters());
        p.addAll(head.parameters());
        return p;
    }

    /** The block-causal bias over {@code [noisy; clean]}: 0 where attention is allowed. */
    private Tensor trainingMask() {
        int n = 2 * length;
        float[] bias = new float[n * n];
        for (int i = 0; i < n; i++) {
            boolean noisyQuery = i < length;
            int bi = (i % length) / block;
            for (int j = 0; j < n; j++) {
                boolean noisyKey = j < length;
                int bj = (j % length) / block;
                boolean allowed = noisyQuery ? (noisyKey ? bi == bj : bj < bi) : !noisyKey && bj <= bi;
                bias[i * n + j] = allowed ? 0 : -1e9f;
            }
        }
        return Tensor.of(Shape.of(n, n), bias);
    }

    private Var embed(int[][] tokens, int[] positionOf) {
        int n = tokens.length;
        int t = tokens[0].length;
        float[] oneHot = new float[n * t * vocabulary];
        float[] pos = new float[n * t * length];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < t; j++) {
                oneHot[(i * t + j) * vocabulary + tokens[i][j]] = 1;
                pos[(i * t + j) * length + positionOf[j]] = 1;
            }
        }
        return Var.of(Tensor.of(Shape.of(n * t, vocabulary), oneHot)).matmul(embedding)
                .add(Var.of(Tensor.of(Shape.of(n * t, length), pos)).matmul(positions));
    }

    private Var split(Var x, int n, int t) {
        return x.reshape(n, t, heads, width / heads).permute(0, 2, 1, 3);
    }

    /**
     * Runs the transformer on {@code [n * t, width]} hidden states.
     *
     * @param cacheK per layer, cached keys {@code [n, heads, past, dh]} to prepend, or null
     * @param newK when not null, receives each layer's keys of these tokens
     */
    private Var run(Var x, int n, int t, Tensor bias, Tensor[] cacheK, Tensor[] cacheV, Tensor[] newK,
            Tensor[] newV) {
        float scale = (float) (1 / Math.sqrt(width / heads));
        for (int l = 0; l < layers.size(); l++) {
            Layer layer = layers.get(l);
            Var h = layer.norm1().forward(x);
            Var q = split(layer.q().forward(h), n, t);
            Var k = split(layer.k().forward(h), n, t);
            Var v = split(layer.v().forward(h), n, t);
            if (newK != null) {
                newK[l] = k.value();
                newV[l] = v.value();
            }
            if (cacheK != null && cacheK[l] != null) {
                k = Var.concat(2, Var.of(cacheK[l]), k);
                v = Var.concat(2, Var.of(cacheV[l]), v);
            }
            Var a = Functional.attention(q, k, v, bias, scale, 0, null).permute(0, 2, 1, 3).reshape(n * t, width);
            x = x.add(layer.o().forward(a));
            x = x.add(layer.down().forward(layer.up().forward(layer.norm2().forward(x))));
        }
        return head.forward(finalNorm.forward(x));
    }

    /**
     * The NELBO training loss, per token.
     *
     * @param tokens clean sequences of {@code length} tokens, padded with the end token
     * @param random the source of mask ratios and masks
     * @return the loss
     */
    public Var loss(int[][] tokens, RandomGenerator random) {
        int n = tokens.length;
        int[][] input = new int[n][2 * length];
        float[] weight = new float[n * length * vocabulary];
        for (int i = 0; i < n; i++) {
            for (int b = 0; b < length / block; b++) {
                int k = 1 + random.nextInt(block);
                int[] order = shuffled(block, random);
                boolean[] masked = new boolean[block];
                for (int m = 0; m < k; m++) {
                    masked[order[m]] = true;
                }
                for (int j = 0; j < block; j++) {
                    int p = b * block + j;
                    input[i][p] = masked[j] ? maskId : tokens[i][p];
                    if (masked[j]) {
                        weight[(i * length + p) * vocabulary + tokens[i][p]] = (float) ((double) block / k
                                / (length * n));
                    }
                }
            }
            System.arraycopy(tokens[i], 0, input[i], length, length);
        }
        int[] positionOf = new int[2 * length];
        for (int j = 0; j < positionOf.length; j++) {
            positionOf[j] = j % length;
        }
        Var logits = run(embed(input, positionOf), n, 2 * length, trainingBias, null, null, null, null)
                .reshape(n, 2 * length, vocabulary).slice(1, 0, length).reshape(n * length, vocabulary);
        Var logp = Activation.LOG_SOFTMAX.forward(logits);
        return logp.mul(Tensor.of(Shape.of(n * length, vocabulary), weight)).sum().neg();
    }

    /**
     * A Monte Carlo estimate of the NELBO per token on held-out sequences.
     *
     * @param tokens the sequences
     * @param draws mask draws to average
     * @param random the mask source
     * @return nats per token
     */
    public double nelbo(int[][] tokens, int draws, RandomGenerator random) {
        double sum = 0;
        for (int d = 0; d < draws; d++) {
            sum += loss(tokens, random).value().item();
        }
        return sum / draws;
    }

    /**
     * Generated sequences and the decoding cost.
     *
     * @param tokens the sequences, cut after their end token
     * @param tokensPerStep generated tokens (end token included) per forward pass of the
     *     denoiser, the clean-block cache passes not counted
     */
    public record Samples(List<int[]> tokens, double tokensPerStep) {
    }

    /**
     * Generates block by block with a KV cache, unmasking the most confident tokens of each
     * block over {@code steps} denoising passes and stopping after the block that holds the
     * end token.
     *
     * @param count how many sequences
     * @param steps denoising passes per block, at most the block size
     * @param sampling token choice
     * @param random the random source
     * @return the samples
     */
    public Samples sample(int count, int steps, Sampling sampling, RandomGenerator random) {
        int[][] seq = new int[count][length];
        boolean[] done = new boolean[count];
        int[] generated = new int[count];
        long passes = 0;
        Tensor[] cacheK = new Tensor[layers.size()];
        Tensor[] cacheV = new Tensor[layers.size()];
        for (int b = 0; b < length / block; b++) {
            int[][] cur = new int[count][block];
            for (int[] row : cur) {
                Arrays.fill(row, maskId);
            }
            int[] positionOf = new int[block];
            for (int j = 0; j < block; j++) {
                positionOf[j] = b * block + j;
            }
            int active = 0;
            for (boolean d : done) {
                active += d ? 0 : 1;
            }
            for (int s = 1; s <= steps; s++) {
                float[] logits = run(embed(cur, positionOf), count, block, null, cacheK, cacheV, null, null).value()
                        .toArray();
                passes += active;
                for (int i = 0; i < count; i++) {
                    unmask(cur[i], logits, i, s, steps, sampling, random);
                }
            }
            Tensor[] newK = new Tensor[layers.size()];
            Tensor[] newV = new Tensor[layers.size()];
            run(embed(cur, positionOf), count, block, null, cacheK, cacheV, newK, newV);
            for (int l = 0; l < layers.size(); l++) {
                cacheK[l] = cacheK[l] == null ? newK[l] : Tensor.concat(2, cacheK[l], newK[l]);
                cacheV[l] = cacheV[l] == null ? newV[l] : Tensor.concat(2, cacheV[l], newV[l]);
            }
            boolean all = true;
            for (int i = 0; i < count; i++) {
                if (done[i]) {
                    continue;
                }
                System.arraycopy(cur[i], 0, seq[i], b * block, block);
                for (int j = 0; j < block && !done[i]; j++) {
                    generated[i]++;
                    done[i] = cur[i][j] == eosId;
                }
                all &= done[i];
            }
            if (all) {
                break;
            }
        }
        List<int[]> out = new ArrayList<>();
        long total = 0;
        for (int i = 0; i < count; i++) {
            out.add(Arrays.copyOf(seq[i], generated[i]));
            total += generated[i];
        }
        return new Samples(out, (double) total / passes);
    }

    private void unmask(int[] row, float[] logits, int i, int s, int steps, Sampling sampling,
            RandomGenerator random) {
        List<int[]> candidates = new ArrayList<>();
        List<Double> confidence = new ArrayList<>();
        int masked = 0;
        for (int j = 0; j < block; j++) {
            if (row[j] != maskId) {
                continue;
            }
            masked++;
            float[] l = Arrays.copyOfRange(logits, (i * block + j) * vocabulary, (i * block + j + 1) * vocabulary);
            l[maskId] = Float.NEGATIVE_INFINITY;
            int token = sampling.sample(l, new int[0], random);
            candidates.add(new int[] {j, token});
            confidence.add(probability(l, token));
        }
        int keepMasked = (int) Math.round((double) masked * (steps - s) / (steps - s + 1));
        Integer[] order = new Integer[candidates.size()];
        for (int k = 0; k < order.length; k++) {
            order[k] = k;
        }
        Arrays.sort(order, (a, c) -> Double.compare(confidence.get(c), confidence.get(a)));
        for (int k = 0; k < masked - keepMasked; k++) {
            int[] pick = candidates.get(order[k]);
            row[pick[0]] = pick[1];
        }
    }

    private static double probability(float[] logits, int token) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double sum = 0;
        for (float v : logits) {
            sum += Math.exp(v - max);
        }
        return Math.exp(logits[token] - max) / sum;
    }

    private static int[] shuffled(int n, RandomGenerator random) {
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
}
