package io.github.zzuegg.brainiac.experimental.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.VectorQuantizer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * IRIS, a discrete-token transformer world model (Micheli, Alonso and Fleuret, ICLR 2023,
 * <a href="https://arxiv.org/abs/2209.00588">arXiv:2209.00588</a>). A discrete autoencoder turns
 * every observation frame into {@code K} tokens of a {@link VectorQuantizer} codebook; a GPT-style
 * causal transformer reads the interleaved history {@code z_0, a_0, z_1, a_1, ...} (frame tokens
 * and action tokens) and predicts, token by token, the next frame's tokens, and at every action
 * token the reward and whether the episode ends. Imagination ({@link #imagine}) appends actions
 * and samples the next frames' tokens autoregressively, so the model conditions on the whole
 * history in its context window, not just the current frame: it can remember what the frame no
 * longer shows. Behaviour is then learned or planned entirely in imagination; {@link #plan} does
 * random-shooting model-predictive control on the imagined rewards.
 *
 * <p>The sequence for {@code n} steps is {@code n (K + 1) + K} tokens long (the frame after the
 * last action included); the vocabulary is the codebook followed by the actions.
 */
public final class TokenWorldModel {

    /**
     * A recorded stretch of an episode.
     *
     * @param observations {@code n + 1} frames
     * @param actions {@code n} actions
     * @param rewards {@code n} rewards
     * @param dones {@code n} termination flags
     */
    public record Segment(float[][] observations, int[] actions, float[] rewards, boolean[] dones) {
    }

    /**
     * An imagined continuation.
     *
     * @param frames the imagined frame tokens {@code [n][h][K]}
     * @param rewards the predicted rewards {@code [n][h]}
     * @param continuation the predicted probability that the episode goes on {@code [n][h]}
     */
    public record Imagination(int[][][] frames, float[][] rewards, float[][] continuation) {
    }

    private final int observations;
    private final int actions;
    private final int tokens;
    private final int codes;
    private final int codeDim;
    private final int steps;
    private final int length;
    private final SimpleMlp encoder;
    private final SimpleMlp decoder;
    private final VectorQuantizer quantizer;
    private final Model transformer;
    private final Optimizer tokenizerOptimizer = Optimizer.adam(3e-3f);
    private final Optimizer dynamicsOptimizer = Optimizer.adam(1e-3f);

    private TokenWorldModel(int observations, int actions, int tokens, int codes, int width, int heads, int depth,
            int steps, long seed) {
        this.observations = observations;
        this.actions = actions;
        this.tokens = tokens;
        this.codes = codes;
        this.codeDim = 8;
        this.steps = steps;
        this.length = steps * (tokens + 1) + tokens;
        java.util.function.UnaryOperator<Var> silu = v -> v.mul(v.sigmoid());
        encoder = SimpleMlp.of(seed, silu, observations, 128, tokens * codeDim);
        decoder = SimpleMlp.of(seed + 1, silu, tokens * codeDim, 128, observations);
        quantizer = VectorQuantizer.of(codes, codeDim, 0.25f, seed + 2);
        Model.Builder b = Model.sequential().input(length).embedding(codes + actions, width).positionalEncoding();
        for (int l = 0; l < depth; l++) {
            b = b.transformer(heads, true);
        }
        transformer = b.dense(codes + 2).seed(seed + 3).build();
    }

    /**
     * Creates a model.
     *
     * @param observations the frame size
     * @param actions the number of discrete actions
     * @param tokens tokens per frame {@code K}
     * @param codes the codebook size
     * @param width the transformer width
     * @param heads attention heads
     * @param depth transformer blocks
     * @param steps the steps of history in the context window
     * @param seed the seed
     * @return the model
     */
    public static TokenWorldModel of(int observations, int actions, int tokens, int codes, int width, int heads,
            int depth, int steps, long seed) {
        return new TokenWorldModel(observations, actions, tokens, codes, width, heads, depth, steps, seed);
    }

    /** @return tokens per frame */
    public int tokensPerFrame() {
        return tokens;
    }

    /** @return the steps of history the transformer sees */
    public int contextSteps() {
        return steps;
    }

    private static Tensor rows(List<float[]> rows) {
        int n = rows.size();
        int d = rows.get(0).length;
        float[] f = new float[n * d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(rows.get(i), 0, f, i * d, d);
        }
        return Tensor.of(Shape.of(n, d), f);
    }

    /**
     * One step of the frame autoencoder: binary cross-entropy of the reconstruction plus the
     * codebook and commitment terms.
     *
     * @param frames a batch of frames with values in {@code [0, 1]}
     * @return the loss
     */
    public float trainTokenizer(List<float[]> frames) {
        Tensor x = rows(frames);
        int n = frames.size();
        Var z = encoder.forward(Var.of(x)).reshape(n * tokens, codeDim);
        VectorQuantizer.Result q = quantizer.quantize(z);
        Var logits = decoder.forward(q.output().reshape(n, tokens * codeDim));
        Var bce = logits.softplus().sub(logits.mul(x)).mean();
        Var loss = bce.add(q.loss());
        float v = loss.value().item();
        loss.backward();
        List<Var> p = new ArrayList<>(encoder.parameters());
        p.addAll(decoder.parameters());
        p.addAll(quantizer.parameters());
        tokenizerOptimizer.step(p);
        return v;
    }

    /**
     * Re-seeds unused codes from current frame latents.
     *
     * @param frames frames to draw latents from
     * @return the codes restarted
     */
    public int restartDeadCodes(List<float[]> frames) {
        int n = frames.size();
        Tensor z = Var.noGrad(() -> encoder.forward(Var.of(rows(frames)))).value().reshape(n * tokens, codeDim);
        return quantizer.restartDeadCodes(z);
    }

    /**
     * Tokenizes frames.
     *
     * @param frames the frames
     * @return {@code [n][K]} token ids
     */
    public int[][] tokenize(List<float[]> frames) {
        int n = frames.size();
        Tensor z = Var.noGrad(() -> encoder.forward(Var.of(rows(frames)))).value().reshape(n * tokens, codeDim);
        int[] c = quantizer.nearest(z);
        int[][] out = new int[n][tokens];
        for (int i = 0; i < n; i++) {
            System.arraycopy(c, i * tokens, out[i], 0, tokens);
        }
        return out;
    }

    /**
     * Decodes frame tokens.
     *
     * @param frameTokens {@code [n][K]}
     * @return {@code [n][observations]} probabilities
     */
    public float[][] decode(int[][] frameTokens) {
        int n = frameTokens.length;
        int[] flat = new int[n * tokens];
        for (int i = 0; i < n; i++) {
            System.arraycopy(frameTokens[i], 0, flat, i * tokens, tokens);
        }
        float[] p = Var.noGrad(() -> decoder.forward(quantizer.lookup(flat).reshape(n, tokens * codeDim)).sigmoid())
                .value().toArray();
        float[][] out = new float[n][observations];
        for (int i = 0; i < n; i++) {
            System.arraycopy(p, i * observations, out[i], 0, observations);
        }
        return out;
    }

    /**
     * One step of the dynamics transformer on segments of at most {@link #contextSteps} steps:
     * cross-entropy of every next frame token, squared error of the rewards and binary
     * cross-entropy of the terminations.
     *
     * @param segments the segments
     * @return the loss
     */
    public float trainDynamics(List<Segment> segments) {
        int b = segments.size();
        int vocab = codes + 2;
        float[] input = new float[b * length];
        float[] codeWeight = new float[b * length * vocab];
        float[] rewardTarget = new float[b * length];
        float[] doneTarget = new float[b * length];
        float[] actionWeight = new float[b * length];
        int scoredCodes = 0;
        int scoredActions = 0;
        for (int i = 0; i < b; i++) {
            Segment s = segments.get(i);
            int n = Math.min(steps, s.actions().length);
            int[][] z = tokenize(java.util.Arrays.asList(s.observations()).subList(0, n + 1));
            for (int t = 0; t <= n; t++) {
                int base = t * (tokens + 1);
                for (int k = 0; k < tokens; k++) {
                    input[i * length + base + k] = z[t][k];
                    if (k + 1 < tokens) {
                        codeWeight[(i * length + base + k) * vocab + z[t][k + 1]] = 1;
                        scoredCodes++;
                    }
                }
                if (t < n) {
                    int ap = base + tokens;
                    input[i * length + ap] = codes + s.actions()[t];
                    codeWeight[(i * length + ap) * vocab + z[t + 1][0]] = 1;
                    scoredCodes++;
                    rewardTarget[i * length + ap] = s.rewards()[t];
                    doneTarget[i * length + ap] = s.dones()[t] ? 1 : 0;
                    actionWeight[i * length + ap] = 1;
                    scoredActions++;
                }
            }
        }
        Var out = transformer.forward(Var.of(Tensor.of(Shape.of(b, length), input)), true).reshape(b * length, vocab);
        Var codeLogits = out.slice(1, 0, codes);
        Var logp = Activation.LOG_SOFTMAX.forward(codeLogits);
        float[] cw = new float[b * length * codes];
        for (int r = 0; r < b * length; r++) {
            System.arraycopy(codeWeight, r * vocab, cw, r * codes, codes);
        }
        Var ce = logp.mul(Tensor.of(Shape.of(b * length, codes), cw)).sum().neg().mul(1f / Math.max(1, scoredCodes));
        Tensor aw = Tensor.of(Shape.of(b * length, 1), actionWeight);
        Var reward = out.slice(1, codes, codes + 1);
        Var done = out.slice(1, codes + 1, codes + 2);
        float inv = 1f / Math.max(1, scoredActions);
        Var rl = reward.sub(Tensor.of(Shape.of(b * length, 1), rewardTarget)).square().mul(aw).sum().mul(inv);
        Var dl = done.softplus().sub(done.mul(Tensor.of(Shape.of(b * length, 1), doneTarget))).mul(aw).sum().mul(inv);
        Var loss = ce.add(rl.mul(10f)).add(dl);
        float v = loss.value().item();
        loss.backward();
        dynamicsOptimizer.step(transformer.parameters());
        return v;
    }

    /**
     * The token history of an episode so far, for {@link #imagine}.
     *
     * @param frames the frames seen
     * @param pastActions the actions taken between them ({@code frames - 1})
     * @return the interleaved token sequence ending with the last frame's tokens
     */
    public int[] history(List<float[]> frames, List<Integer> pastActions) {
        int[][] z = tokenize(frames);
        int[] seq = new int[frames.size() * (tokens + 1) - 1];
        for (int t = 0; t < frames.size(); t++) {
            System.arraycopy(z[t], 0, seq, t * (tokens + 1), tokens);
            if (t < pastActions.size()) {
                seq[t * (tokens + 1) + tokens] = codes + pastActions.get(t);
            }
        }
        return seq;
    }

    /**
     * Imagines continuations of a history under action sequences.
     *
     * @param history the token history (from {@link #history}), the same for every sequence
     * @param plans the action sequences {@code [n][h]}
     * @param greedy whether to take the most likely next token instead of sampling
     * @param random the sampling source
     * @return the imagined frames, rewards and continuation probabilities
     */
    public Imagination imagine(int[] history, int[][] plans, boolean greedy, RandomGenerator random) {
        int n = plans.length;
        int h = plans[0].length;
        int vocab = codes + 2;
        List<int[]> seqs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            seqs.add(history.clone());
        }
        int[][][] frames = new int[n][h][tokens];
        float[][] rewards = new float[n][h];
        float[][] cont = new float[n][h];
        for (int step = 0; step < h; step++) {
            for (int i = 0; i < n; i++) {
                int[] s = seqs.get(i);
                int[] t = java.util.Arrays.copyOf(s, s.length + 1);
                t[s.length] = codes + plans[i][step];
                seqs.set(i, trim(t));
            }
            for (int k = 0; k <= tokens; k++) {
                float[] logits = forwardLast(seqs, vocab);
                for (int i = 0; i < n; i++) {
                    int o = i * vocab;
                    if (k == 0) {
                        rewards[i][step] = logits[o + codes];
                        cont[i][step] = (float) (1 / (1 + Math.exp(logits[o + codes + 1])));
                    }
                    if (k == tokens) {
                        continue;
                    }
                    int token = greedy ? argmax(logits, o, codes) : sample(logits, o, codes, random);
                    frames[i][step][k] = token;
                    int[] s = seqs.get(i);
                    int[] t = java.util.Arrays.copyOf(s, s.length + 1);
                    t[s.length] = token;
                    seqs.set(i, trim(t));
                }
                if (k == tokens - 1) {
                    break;
                }
            }
        }
        return new Imagination(frames, rewards, cont);
    }

    /** Drops whole steps from the front until the sequence fits the context. */
    private int[] trim(int[] seq) {
        int over = seq.length - length;
        if (over <= 0) {
            return seq;
        }
        int drop = ((over + tokens) / (tokens + 1)) * (tokens + 1);
        return java.util.Arrays.copyOfRange(seq, drop, seq.length);
    }

    /** Logits at the last position of each sequence (sequences padded to the context length). */
    private float[] forwardLast(List<int[]> seqs, int vocab) {
        int n = seqs.size();
        float[] input = new float[n * length];
        int[] last = new int[n];
        for (int i = 0; i < n; i++) {
            int[] s = seqs.get(i);
            for (int j = 0; j < s.length; j++) {
                input[i * length + j] = s[j];
            }
            last[i] = s.length - 1;
        }
        float[] out = Var.noGrad(() -> transformer.forward(Var.of(Tensor.of(Shape.of(n, length), input)), false))
                .value().toArray();
        float[] res = new float[n * vocab];
        for (int i = 0; i < n; i++) {
            System.arraycopy(out, (i * length + last[i]) * vocab, res, i * vocab, vocab);
        }
        return res;
    }

    private static int argmax(float[] v, int offset, int n) {
        int best = 0;
        for (int i = 1; i < n; i++) {
            if (v[offset + i] > v[offset + best]) {
                best = i;
            }
        }
        return best;
    }

    private static int sample(float[] v, int offset, int n, RandomGenerator random) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, v[offset + i]);
        }
        double sum = 0;
        double[] p = new double[n];
        for (int i = 0; i < n; i++) {
            p[i] = Math.exp(v[offset + i] - max);
            sum += p[i];
        }
        double u = random.nextDouble() * sum;
        for (int i = 0; i < n; i++) {
            u -= p[i];
            if (u <= 0) {
                return i;
            }
        }
        return n - 1;
    }

    /**
     * Random-shooting model-predictive control in imagination: imagine {@code candidates}
     * random action sequences of length {@code horizon} from the history and return the first
     * action of the one with the highest predicted discounted return.
     *
     * @param history the token history
     * @param candidates action sequences to try
     * @param horizon their length
     * @param gamma the discount
     * @param random the random source
     * @return the action to take
     */
    public int plan(int[] history, int candidates, int horizon, float gamma, RandomGenerator random) {
        int[][] plans = new int[candidates][horizon];
        for (int[] p : plans) {
            for (int t = 0; t < horizon; t++) {
                p[t] = random.nextInt(actions);
            }
        }
        Imagination im = imagine(history, plans, true, random);
        int best = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < candidates; i++) {
            double value = 0;
            double alive = 1;
            double discount = 1;
            for (int t = 0; t < horizon; t++) {
                value += discount * alive * im.rewards()[i][t];
                alive *= im.continuation()[i][t];
                discount *= gamma;
            }
            if (value > bestValue) {
                bestValue = value;
                best = i;
            }
        }
        return plans[best][0];
    }
}
