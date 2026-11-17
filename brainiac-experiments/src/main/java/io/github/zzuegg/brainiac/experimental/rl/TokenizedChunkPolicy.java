package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ActionTokenizer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * An autoregressive, VLA-style policy: a small causal transformer (core
 * {@link TransformerBlock}s) reads the observation as tokens (each value binned) and then
 * writes the action chunk as tokens of an {@link ActionTokenizer}, ending with an
 * end-of-chunk token, as RT-2, OpenVLA and pi0-FAST do. Sampling the tokens one by one
 * represents multimodal action distributions without any special head.
 *
 * <p>Sequence layout: {@code [obs tokens, BOS, action tokens, EOS, PAD...]}; the loss is the
 * cross-entropy of the action tokens and the EOS only.
 */
public final class TokenizedChunkPolicy implements ChunkPolicy, Agent {
    private final ActionTokenizer tokenizer;
    private final int observationSize;
    private final int observationBins;
    private final float observationRange;
    private final int maxTokens;
    private final int bos;
    private final int eos;
    private final int pad;
    private final int vocabulary;
    private final int length;
    private final Model model;
    private final Optimizer optimizer = Optimizer.adam(3e-3f);
    private final SplittableRandom random;
    private float temperature = 0.7f;
    private int execute;
    private Agent executor;

    private TokenizedChunkPolicy(ActionTokenizer tokenizer, int observationSize, int observationBins,
            float observationRange, int maxTokens, int width, int layers, long seed) {
        this.tokenizer = tokenizer;
        this.observationSize = observationSize;
        this.observationBins = observationBins;
        this.observationRange = observationRange;
        this.maxTokens = maxTokens;
        int a = tokenizer.size();
        this.bos = a;
        this.eos = a + 1;
        this.pad = a + 2;
        this.vocabulary = a + 3 + observationSize * observationBins;
        this.length = observationSize + maxTokens + 1;
        Model.Builder b = Model.sequential().input(length).embedding(vocabulary, width).positionalEncoding();
        for (int i = 0; i < layers; i++) {
            b.add(TransformerBlock.of(2).causal(true));
        }
        this.model = b.dense(vocabulary).seed(seed).build();
        this.random = new SplittableRandom(seed + 1);
        this.execute = Math.max(1, tokenizer.horizon() / 2);
    }

    /**
     * Creates a policy.
     *
     * @param tokenizer the action tokenizer
     * @param observationSize the observation size
     * @param observationBins bins per observation value
     * @param observationRange observations are binned over {@code [-range, range]}
     * @param maxTokens the most action tokens per chunk
     * @param width the transformer width
     * @param layers the number of transformer blocks
     * @param seed the seed
     * @return the policy
     */
    public static TokenizedChunkPolicy of(ActionTokenizer tokenizer, int observationSize, int observationBins,
            float observationRange, int maxTokens, int width, int layers, long seed) {
        return new TokenizedChunkPolicy(tokenizer, observationSize, observationBins, observationRange, maxTokens,
                width, layers, seed);
    }

    /**
     * Sets the sampling temperature.
     *
     * @param t the temperature, 0 for greedy
     * @return this policy
     */
    public TokenizedChunkPolicy temperature(float t) {
        this.temperature = t;
        return this;
    }

    /**
     * Sets how many actions of each chunk run before replanning.
     *
     * @param steps 1 to the horizon
     * @return this policy
     */
    public TokenizedChunkPolicy execute(int steps) {
        this.execute = steps;
        this.executor = null;
        return this;
    }

    private int[] prefix(float[] observation) {
        int[] p = new int[observationSize + 1];
        for (int i = 0; i < observationSize; i++) {
            double u = (Math.max(-observationRange, Math.min(observationRange, observation[i])) + observationRange)
                    / (2 * observationRange);
            int bin = (int) Math.min(observationBins - 1, Math.floor(u * observationBins));
            p[i] = pad + 1 + i * observationBins + bin;
        }
        p[observationSize] = bos;
        return p;
    }

    /**
     * Trains on (observation, chunk) pairs.
     *
     * @param pairs the demonstrations, cut into chunks of the tokenizer's horizon
     * @param steps optimizer steps
     * @param batch the batch size
     * @return the last loss
     */
    public float train(ActionChunking.Pairs pairs, int steps, int batch) {
        int[][] encoded = new int[pairs.size()][];
        for (int i = 0; i < encoded.length; i++) {
            encoded[i] = tokenizer.encodeChunk(ActionChunking.unflatten(pairs.chunks()[i], tokenizer.actionSize()));
        }
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            float[] in = new float[batch * length];
            float[] target = new float[batch * length * vocabulary];
            int counted = 0;
            for (int b = 0; b < batch; b++) {
                int k = random.nextInt(pairs.size());
                int[] seq = sequence(prefix(pairs.observations()[k]), encoded[k]);
                for (int j = 0; j < length; j++) {
                    in[b * length + j] = seq[j];
                    int next = seq[j + 1];
                    if (j >= observationSize && next != pad) {
                        target[(b * length + j) * vocabulary + next] = 1;
                        counted++;
                    }
                }
            }
            Var logits = model.forward(Var.of(Tensor.of(Shape.of(batch, length), in)), true);
            Var logp = Activation.LOG_SOFTMAX.forward(logits.reshape(batch * length, vocabulary));
            Var loss = logp.mul(Tensor.of(Shape.of(batch * length, vocabulary), target)).sum().neg()
                    .div((float) counted);
            last = loss.value().item();
            loss.backward();
            RlTensors.clip(model.parameters(), 1f);
            optimizer.step(model.parameters());
        }
        return last;
    }

    private int[] sequence(int[] prefix, int[] tokens) {
        int[] seq = new int[length + 1];
        Arrays.fill(seq, pad);
        System.arraycopy(prefix, 0, seq, 0, prefix.length);
        int n = Math.min(tokens.length, maxTokens);
        System.arraycopy(tokens, 0, seq, prefix.length, n);
        seq[prefix.length + n] = eos;
        return seq;
    }

    /**
     * Samples the action tokens for one observation.
     *
     * @param observation the observation
     * @param random the sampling noise
     * @return the tokens, without BOS and EOS
     */
    public int[] sampleTokens(float[] observation, SplittableRandom random) {
        int[] prefix = prefix(observation);
        float[] in = new float[length];
        Arrays.fill(in, pad);
        for (int i = 0; i < prefix.length; i++) {
            in[i] = prefix[i];
        }
        List<Integer> out = new ArrayList<>();
        for (int pos = prefix.length - 1; pos < length - 1 && out.size() < maxTokens; pos++) {
            float[] all = model.forward(Var.of(Tensor.of(Shape.of(1, length), in.clone())), false).value().toArray();
            float[] row = Arrays.copyOfRange(all, pos * vocabulary, pos * vocabulary + tokenizer.size() + 2);
            row[bos] = Float.NEGATIVE_INFINITY;
            int tok = temperature <= 0 ? RlTensors.argmax(row) : sample(row, random);
            if (tok == eos) {
                break;
            }
            out.add(tok);
            in[pos + 1] = tok;
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private int sample(float[] logits, SplittableRandom random) {
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) {
            max = Math.max(max, l);
        }
        double[] p = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp((logits[i] - max) / temperature);
            sum += p[i];
        }
        double u = random.nextDouble() * sum;
        for (int i = 0; i < p.length; i++) {
            u -= p[i];
            if (u <= 0) {
                return i;
            }
        }
        return p.length - 1;
    }

    @Override
    public float[][] chunk(float[] observation, SplittableRandom random) {
        float[][] c = tokenizer.decodeChunk(sampleTokens(observation, random));
        for (int i = 0; i < c.length; i++) {
            c[i] = ActionChunking.clip(c[i]);
        }
        return c;
    }

    @Override
    public int horizon() {
        return tokenizer.horizon();
    }

    @Override
    public int actionSize() {
        return tokenizer.actionSize();
    }

    /** @return the transformer */
    public Model model() {
        return model;
    }

    @Override
    public void startEpisode() {
        executor().startEpisode();
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return executor().act(observation, explore);
    }

    private Agent executor() {
        if (executor == null) {
            executor = ActionChunking.recedingHorizon(this, execute, random.nextLong());
        }
        return executor;
    }
}
