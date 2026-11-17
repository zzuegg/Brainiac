package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * A small pre-norm decoder-only transformer (GPT shape, rotary positions) whose every part
 * can be reached from outside, for the language-model techniques that need more than
 * {@link LanguageModel#next}: the residual stream after each block (DoLa, early exit,
 * contrastive search, ShortGPT block importance), replaceable projections (LoRA, VeRA,
 * (IA)^3, PiSSA, DoRA, pruning, low-rank factorization), learned prefixes per layer (prefix
 * tuning), a soft prompt (prompt tuning), bottleneck adapters after attention and the
 * feed-forward network (Houlsby adapters), skipped blocks (layer skipping), explicit
 * positions with a scale (position interpolation) and document-masked attention (sequence
 * packing).
 *
 * <p>It implements {@link LanguageModel} with a state that keeps the tokens and recomputes
 * the whole sequence on each step, which is exact and simple for the short sequences of
 * experiments, and supports {@code truncate} and {@code select}, so the core
 * {@code Generator} (including speculative decoding and beam search) drives it unchanged.
 *
 * <p>The projections are core {@link Dense} layers named {@code block.<i>.attn.q|k|v|o}
 * and {@code block.<i>.ffn.up|down}; {@link #replaceLinears} swaps any of them for another
 * {@link Layer}, so an adapter wrapping a {@code Dense} is a one-line change.
 */
public final class HookedDecoder implements LanguageModel {

    /**
     * The shape of a decoder.
     *
     * @param vocab the vocabulary size
     * @param dim the residual width
     * @param layers the number of blocks
     * @param heads attention heads; {@code dim / heads} must be even
     * @param ffnMultiplier the feed-forward width as a multiple of {@code dim}
     * @param ropeBase the rotary base
     */
    public record Settings(int vocab, int dim, int layers, int heads, int ffnMultiplier, float ropeBase) {

        /**
         * Checks the settings.
         *
         * @param vocab vocabulary size
         * @param dim width
         * @param layers blocks
         * @param heads heads
         * @param ffnMultiplier feed-forward multiple
         * @param ropeBase rotary base
         */
        public Settings {
            if (vocab < 2 || dim < 2 || layers < 1 || heads < 1 || dim % heads != 0 || (dim / heads) % 2 != 0
                    || ffnMultiplier < 1 || !(ropeBase > 1)) {
                throw new IllegalArgumentException("bad decoder settings");
            }
        }

        /**
         * @param vocab the vocabulary size
         * @return 32 wide, 2 blocks of 4 heads
         */
        public static Settings tiny(int vocab) {
            return new Settings(vocab, 32, 2, 4, 4, 10000f);
        }

        /**
         * @param n blocks
         * @return a copy with {@code n} blocks
         */
        public Settings withLayers(int n) {
            return new Settings(vocab, dim, n, heads, ffnMultiplier, ropeBase);
        }

        /**
         * @param d width
         * @param h heads
         * @return a copy with that width and head count
         */
        public Settings withDim(int d, int h) {
            return new Settings(vocab, d, layers, h, ffnMultiplier, ropeBase);
        }
    }

    private static final class Block {
        final LayerNorm ln1;
        final LayerNorm ln2;
        final Layer[] linears = new Layer[6];
        Var prefixK;
        Var prefixV;
        Layer attnAdapter;
        Layer ffnAdapter;

        Block(LayerNorm ln1, LayerNorm ln2) {
            this.ln1 = ln1;
            this.ln2 = ln2;
        }
    }

    private static final String[] LINEAR_NAMES = {"attn.q", "attn.k", "attn.v", "attn.o", "ffn.up", "ffn.down"};

    private final Settings settings;
    private final Var embed;
    private final List<Block> blocks = new ArrayList<>();
    private final LayerNorm finalNorm;
    private Layer head;
    private PromptEncoder softPrompt;
    private boolean[] skipped;
    private float positionScale = 1;
    private float embeddingNoise;
    private final SplittableRandom noise;
    private final long seed;

    private HookedDecoder(Settings s, long seed) {
        this.settings = s;
        this.seed = seed;
        SplittableRandom r = new SplittableRandom(seed);
        this.noise = r.split();
        this.embed = Var.param(Tensor.normal(Shape.of(s.vocab(), s.dim()), 0, 0.5f, r));
        int ffn = s.dim() * s.ffnMultiplier();
        for (int i = 0; i < s.layers(); i++) {
            Block b = new Block(LayerNorm.of(s.dim()), LayerNorm.of(s.dim()));
            for (int j = 0; j < 4; j++) {
                b.linears[j] = new Dense(s.dim(), s.dim(), Activation.IDENTITY, Init.XAVIER, r);
            }
            b.linears[4] = new Dense(s.dim(), ffn, Activation.GELU, Init.HE, r);
            Tensor down = Init.XAVIER.create(Shape.of(ffn, s.dim()), ffn, s.dim(), r)
                    .mul((float) (1 / Math.sqrt(2.0 * s.layers())));
            b.linears[5] = new Dense(ffn, s.dim(), Activation.IDENTITY, down, Tensor.zeros(Shape.of(s.dim())));
            blocks.add(b);
        }
        this.finalNorm = LayerNorm.of(s.dim());
        this.head = new Dense(s.dim(), s.vocab(), Activation.IDENTITY, Init.XAVIER, r);
        this.skipped = new boolean[s.layers()];
    }

    /**
     * A randomly initialized decoder.
     *
     * @param settings the shape
     * @param seed the seed
     * @return the decoder
     */
    public static HookedDecoder create(Settings settings, long seed) {
        return new HookedDecoder(settings, seed);
    }

    /**
     * A copy of the base model (embedding, norms, projections, head) with its own weights,
     * without hooks. The projections must still be the original {@code Dense} layers.
     *
     * @return the copy
     */
    public HookedDecoder copy() {
        HookedDecoder c = new HookedDecoder(settings, seed);
        List<Var> from = baseParameters();
        List<Var> to = c.baseParameters();
        if (from.size() != to.size()) {
            throw new IllegalStateException("replaced projections cannot be copied");
        }
        for (int i = 0; i < from.size(); i++) {
            to.get(i).assign(from.get(i).value().add(0f));
        }
        return c;
    }

    /** @return the shape */
    public Settings settings() {
        return settings;
    }

    @Override
    public int vocabSize() {
        return settings.vocab();
    }

    /** @return the number of blocks */
    public int layers() {
        return blocks.size();
    }

    /** @return the token embedding {@code [vocab, dim]} */
    public Var embedding() {
        return embed;
    }

    /** @return the output projection (a {@link Dense} unless replaced) */
    public Layer head() {
        return head;
    }

    /**
     * Replaces the output projection, for example by a trimmed one.
     *
     * @param layer a layer from {@code dim} to the vocabulary size
     */
    public void head(Layer layer) {
        this.head = Objects.requireNonNull(layer);
    }

    // ------------------------------------------------------------------ hooks

    /** @return the projections by name, in a stable order */
    public Map<String, Layer> linears() {
        Map<String, Layer> m = new LinkedHashMap<>();
        for (int i = 0; i < blocks.size(); i++) {
            for (int j = 0; j < LINEAR_NAMES.length; j++) {
                m.put("block." + i + "." + LINEAR_NAMES[j], blocks.get(i).linears[j]);
            }
        }
        return m;
    }

    /**
     * Replaces projections: {@code replace} gets each name and current layer and returns
     * the layer to use (the same one to keep it).
     *
     * @param replace the replacement
     */
    public void replaceLinears(BiFunction<String, Layer, Layer> replace) {
        for (int i = 0; i < blocks.size(); i++) {
            for (int j = 0; j < LINEAR_NAMES.length; j++) {
                Block b = blocks.get(i);
                b.linears[j] = Objects.requireNonNull(replace.apply("block." + i + "." + LINEAR_NAMES[j], b.linears[j]));
            }
        }
    }

    /**
     * Sets learned key and value prefixes of one block (prefix tuning); {@code null} removes
     * them.
     *
     * @param layer the block
     * @param keys {@code [prefix, dim]} or null
     * @param values {@code [prefix, dim]} or null
     */
    public void prefix(int layer, Var keys, Var values) {
        Block b = blocks.get(layer);
        b.prefixK = keys;
        b.prefixV = values;
    }

    /**
     * Makes the soft prompt of prompt tuning and P-tuning: vectors read before the tokens.
     */
    public interface PromptEncoder {

        /** @return the prompt {@code [length, dim]}, recomputed on every forward pass */
        Var prompt();

        /** @return the trainable values behind it */
        List<Var> parameters();
    }

    /**
     * Sets a soft prompt, learned vectors read before the tokens (prompt tuning); {@code null}
     * removes it.
     *
     * @param prompt {@code [length, dim]} or null
     */
    public void softPrompt(Var prompt) {
        this.softPrompt = prompt == null ? null : new PromptEncoder() {
            @Override
            public Var prompt() {
                return prompt;
            }

            @Override
            public List<Var> parameters() {
                return List.of(prompt);
            }
        };
    }

    /**
     * Sets a soft prompt computed by an encoder (P-tuning); {@code null} removes it.
     *
     * @param encoder the prompt encoder or null
     */
    public void softPrompt(PromptEncoder encoder) {
        this.softPrompt = encoder;
    }

    /**
     * Sets adapters of one block, applied to the attention output and to the feed-forward
     * output before each is added to the residual stream (a Houlsby adapter keeps its own
     * skip connection, so it starts as the identity); {@code null} for none.
     *
     * @param layer the block
     * @param afterAttention the adapter after attention, or null
     * @param afterFfn the adapter after the feed-forward network, or null
     */
    public void adapters(int layer, Layer afterAttention, Layer afterFfn) {
        Block b = blocks.get(layer);
        b.attnAdapter = afterAttention;
        b.ffnAdapter = afterFfn;
    }

    /**
     * Skips blocks: their residual update is left out (layer skipping, LayerDrop at
     * inference, ShortGPT layer removal).
     *
     * @param layers the blocks to skip; none to run all
     */
    public void skip(int... layers) {
        skipped = new boolean[blocks.size()];
        for (int l : layers) {
            skipped[l] = true;
        }
    }

    /** @return whether each block is skipped */
    public boolean[] skipped() {
        return skipped.clone();
    }

    /**
     * Multiplies every rotary position by {@code scale}: {@code 1/s} is position
     * interpolation to an {@code s} times longer context (Chen et al. 2023,
     * <a href="https://arxiv.org/abs/2306.15595">arXiv:2306.15595</a>).
     *
     * @param scale the position scale, 1 for none
     */
    public void positionScale(float scale) {
        this.positionScale = scale;
    }

    /**
     * NEFTune (Jain et al. 2023, <a href="https://arxiv.org/abs/2310.05914">arXiv:2310.05914</a>):
     * while training, adds uniform noise of size {@code alpha / sqrt(steps * dim)} to the
     * input embeddings, which regularizes instruction fine-tuning.
     *
     * @param alpha the noise scale; 0 turns it off
     */
    public void embeddingNoise(float alpha) {
        this.embeddingNoise = alpha;
    }

    // ------------------------------------------------------------------ parameters

    /** @return every trainable value: embedding, norms, projections, head, and any hooks */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(baseParameters());
        p.addAll(hookParameters());
        return p;
    }

    /** @return the parameters of the base model (embedding, norms, projections, head) */
    public List<Var> baseParameters() {
        List<Var> p = new ArrayList<>();
        p.add(embed);
        for (Block b : blocks) {
            p.addAll(b.ln1.parameters());
            p.addAll(b.ln2.parameters());
            for (Layer l : b.linears) {
                p.addAll(l.parameters());
            }
        }
        p.addAll(finalNorm.parameters());
        p.addAll(head.parameters());
        return p;
    }

    /** @return the parameters of prefixes, the soft prompt and adapters */
    public List<Var> hookParameters() {
        List<Var> p = new ArrayList<>();
        if (softPrompt != null) {
            p.addAll(softPrompt.parameters());
        }
        for (Block b : blocks) {
            if (b.prefixK != null) {
                p.add(b.prefixK);
                p.add(b.prefixV);
            }
            if (b.attnAdapter != null) {
                p.addAll(b.attnAdapter.parameters());
            }
            if (b.ffnAdapter != null) {
                p.addAll(b.ffnAdapter.parameters());
            }
        }
        return p;
    }

    /** @return the bias vectors of every projection and the norms' shifts (for BitFit) */
    public List<Var> biases() {
        List<Var> p = new ArrayList<>();
        for (Var v : baseParameters()) {
            if (v.shape().rank() == 1) {
                p.add(v);
            }
        }
        return p;
    }

    /** Clears the gradient of every parameter, trained or not. */
    public void zeroGrad() {
        for (Var v : parameters()) {
            v.zeroGrad();
        }
    }

    // ------------------------------------------------------------------ forward

    /**
     * The result of a forward pass.
     *
     * @param hidden the residual stream {@code [batch, steps, dim]}: index 0 after the
     *     embedding, index {@code i + 1} after block {@code i} (soft-prompt steps removed)
     * @param logits {@code [batch, steps, vocab]} after the last block
     */
    public record Run(List<Var> hidden, Var logits) {
    }

    /**
     * Runs the model with positions {@code 0..steps-1}.
     *
     * @param tokens {@code [batch][steps]}
     * @return the run
     */
    public Run forward(int[][] tokens) {
        return forward(tokens, null, null, false);
    }

    /**
     * Runs the model.
     *
     * @param tokens {@code [batch][steps]}
     * @param positions {@code [batch][steps]} rotary positions, or null for {@code 0..steps-1}
     * @param segments {@code [batch][steps]} document ids: a token attends only to earlier
     *     tokens of the same document (sequence packing); null for one document per row
     * @param training whether to add embedding noise
     * @return the run
     */
    public Run forward(int[][] tokens, int[][] positions, int[][] segments, boolean training) {
        return forward(tokens, positions, segments, training, blocks.size());
    }

    /**
     * Runs the first {@code depth} blocks only (early exit); the logits are those of the
     * last block run.
     *
     * @param tokens {@code [batch][steps]}
     * @param positions positions or null
     * @param segments document ids or null
     * @param training whether to add embedding noise
     * @param depth the blocks to run
     * @return the run
     */
    public Run forward(int[][] tokens, int[][] positions, int[][] segments, boolean training, int depth) {
        Walk w = walk(tokens, positions, segments, training);
        List<Var> hidden = new ArrayList<>();
        hidden.add(w.hidden());
        while (w.depth() < depth) {
            hidden.add(w.advance());
        }
        return new Run(hidden, project(hidden.get(hidden.size() - 1)));
    }

    /**
     * Starts a pass that runs one block at a time, so a caller can stop early (CALM early
     * exit, self-speculative drafting with the first blocks).
     *
     * @param tokens {@code [batch][steps]}
     * @param positions positions or null
     * @param segments document ids or null
     * @param training whether to add embedding noise
     * @return the pass, before the first block
     */
    public Walk walk(int[][] tokens, int[][] positions, int[][] segments, boolean training) {
        return new Walk(tokens, positions, segments, training);
    }

    /** A forward pass that runs one block at a time. */
    public final class Walk {
        private Var x;
        private final Tensor[] rot;
        private final int[][] segments;
        private final int prompt;
        private int depth;

        private Walk(int[][] tokens, int[][] positions, int[][] segments, boolean training) {
            int batch = tokens.length;
            int steps = tokens[0].length;
            int dim = settings.dim();
            int[] flat = new int[batch * steps];
            for (int b = 0; b < batch; b++) {
                System.arraycopy(tokens[b], 0, flat, b * steps, steps);
            }
            Var e = embed.gather(0, ids(flat)).reshape(batch, steps, dim);
            if (training && embeddingNoise > 0) {
                float a = (float) (embeddingNoise / Math.sqrt((double) steps * dim));
                e = e.add(Tensor.uniform(Shape.of(batch, steps, dim), -a, a, noise));
            }
            Var sp = softPrompt == null ? null : softPrompt.prompt();
            this.prompt = sp == null ? 0 : sp.shape().dim(0);
            if (prompt > 0) {
                e = Var.concat(1, sp.reshape(1, prompt, dim).expand(Shape.of(batch, prompt, dim)), e);
            }
            int total = prompt + steps;
            float[][] pos = new float[batch][total];
            for (int b = 0; b < batch; b++) {
                for (int t = 0; t < total; t++) {
                    int p = t < prompt ? t : (positions == null ? t : positions[b][t - prompt] + prompt);
                    pos[b][t] = p * positionScale;
                }
            }
            this.x = e;
            this.rot = rotaryTables(pos);
            this.segments = segments;
        }

        /** @return the blocks run so far */
        public int depth() {
            return depth;
        }

        /** @return the residual stream {@code [batch, steps, dim]} after the blocks run so far */
        public Var hidden() {
            return prompt > 0 ? x.slice(1, prompt, x.shape().dim(1)) : x;
        }

        /**
         * Runs the next block (or passes over it if skipped).
         *
         * @return the residual stream after it
         */
        public Var advance() {
            if (depth >= blocks.size()) {
                throw new IllegalStateException("every block has run");
            }
            if (!skipped[depth]) {
                x = block(blocks.get(depth), x, rot, segments, prompt);
            }
            depth++;
            return hidden();
        }

        /** @return the logits of the residual stream so far, {@code [batch, steps, vocab]} */
        public Var logits() {
            return project(hidden());
        }
    }

    /**
     * The output head on any residual stream: final norm then the vocabulary projection (as
     * early exit and DoLa read intermediate layers).
     *
     * @param hidden {@code [..., dim]}
     * @return logits {@code [..., vocab]}
     */
    public Var project(Var hidden) {
        return head.forward(finalNorm.forward(hidden));
    }

    private static Tensor ids(int[] flat) {
        float[] f = new float[flat.length];
        for (int i = 0; i < flat.length; i++) {
            f[i] = flat[i];
        }
        return Tensor.of(Shape.of(flat.length), f);
    }

    private Tensor[] rotaryTables(float[][] pos) {
        int batch = pos.length;
        int steps = pos[0].length;
        int dh = settings.dim() / settings.heads();
        int half = dh / 2;
        float[] cos = new float[batch * steps * half];
        float[] sin = new float[batch * steps * half];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                for (int i = 0; i < half; i++) {
                    double f = Math.pow(settings.ropeBase(), -2.0 * i / dh);
                    double angle = pos[b][t] * f;
                    cos[(b * steps + t) * half + i] = (float) Math.cos(angle);
                    sin[(b * steps + t) * half + i] = (float) Math.sin(angle);
                }
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(batch, 1, steps, half), cos),
            Tensor.of(Shape.of(batch, 1, steps, half), sin)};
    }

    private static Var rotate(Var x, Tensor[] rot) {
        int d = x.shape().dim(-1);
        int half = d / 2;
        Var x1 = x.slice(-1, 0, half);
        Var x2 = x.slice(-1, half, d);
        return Var.concat(-1, x1.mul(rot[0]).sub(x2.mul(rot[1])), x2.mul(rot[0]).add(x1.mul(rot[1])));
    }

    private Var block(Block b, Var x, Tensor[] rot, int[][] segments, int prompt) {
        int batch = x.shape().dim(0);
        int steps = x.shape().dim(1);
        int dim = settings.dim();
        int heads = settings.heads();
        int dh = dim / heads;
        Var h = b.ln1.forward(x);
        Var q = heads(b.linears[0].forward(h), heads);
        Var k = heads(b.linears[1].forward(h), heads);
        Var v = heads(b.linears[2].forward(h), heads);
        q = rotate(q, rot);
        k = rotate(k, rot);
        int prefix = 0;
        if (b.prefixK != null) {
            prefix = b.prefixK.shape().dim(0);
            Shape ps = Shape.of(batch, heads, prefix, dh);
            Var pk = b.prefixK.reshape(1, prefix, heads, dh).permute(0, 2, 1, 3).expand(ps);
            Var pv = b.prefixV.reshape(1, prefix, heads, dh).permute(0, 2, 1, 3).expand(ps);
            k = Var.concat(2, pk, k);
            v = Var.concat(2, pv, v);
        }
        Tensor bias = mask(batch, steps, prefix, segments, prompt);
        Var a = Functional.attention(q, k, v, bias, (float) (1 / Math.sqrt(dh)), 0, null);
        a = a.permute(0, 2, 1, 3).reshape(batch, steps, dim);
        Var att = b.linears[3].forward(a);
        if (b.attnAdapter != null) {
            att = b.attnAdapter.forward(att);
        }
        x = x.add(att);
        Var f = b.linears[5].forward(b.linears[4].forward(b.ln2.forward(x)));
        if (b.ffnAdapter != null) {
            f = b.ffnAdapter.forward(f);
        }
        return x.add(f);
    }

    private static Var heads(Var x, int heads) {
        Shape s = x.shape();
        return x.reshape(s.dim(0), s.dim(1), heads, s.dim(2) / heads).permute(0, 2, 1, 3);
    }

    private static Tensor mask(int batch, int steps, int prefix, int[][] segments, int prompt) {
        int keys = prefix + steps;
        float[] m = new float[(segments == null ? 1 : batch) * steps * keys];
        for (int b = 0; b < (segments == null ? 1 : batch); b++) {
            for (int i = 0; i < steps; i++) {
                for (int j = 0; j < steps; j++) {
                    boolean blocked = j > i;
                    if (!blocked && segments != null && i >= prompt && j >= prompt) {
                        blocked = segments[b][i - prompt] != segments[b][j - prompt];
                    }
                    if (blocked) {
                        m[(b * steps + i) * keys + prefix + j] = -1e9f;
                    }
                }
            }
        }
        return Tensor.of(Shape.of(segments == null ? 1 : batch, 1, steps, keys), m);
    }

    // ------------------------------------------------------------------ training helpers

    /**
     * Next-token cross-entropy of sequences: every token predicts the one after it.
     *
     * @param tokens {@code [batch][steps]}
     * @param training whether to add embedding noise
     * @return the mean loss
     */
    public Var loss(int[][] tokens, boolean training) {
        int batch = tokens.length;
        int n = tokens[0].length - 1;
        int[][] in = new int[batch][];
        int[][] target = new int[batch][];
        for (int b = 0; b < batch; b++) {
            in[b] = Arrays.copyOf(tokens[b], n);
            target[b] = Arrays.copyOfRange(tokens[b], 1, n + 1);
        }
        return loss(forward(in, null, null, training).logits(), target);
    }

    /**
     * Cross-entropy of logits against targets, skipping negative targets.
     *
     * @param logits {@code [batch, steps, vocab]}
     * @param targets {@code [batch][steps]}, negative to skip
     * @return the mean loss over the counted targets
     */
    public static Var loss(Var logits, int[][] targets) {
        int batch = targets.length;
        int steps = targets[0].length;
        int vocab = logits.shape().dim(-1);
        float[] t = new float[batch * steps];
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < steps; s++) {
                t[b * steps + s] = targets[b][s];
            }
        }
        return Functional.crossEntropy(logits.reshape(batch * steps, vocab), Tensor.of(Shape.of(batch * steps), t));
    }

    /**
     * Log-probabilities of the given next tokens: {@code [batch, steps]}.
     *
     * @param logits {@code [batch, steps, vocab]}
     * @param targets {@code [batch][steps]} token ids
     * @return the log-probability of each target, differentiable
     */
    public static Var tokenLogProbs(Var logits, int[][] targets) {
        int batch = targets.length;
        int steps = targets[0].length;
        int vocab = logits.shape().dim(-1);
        float[] oneHot = new float[batch * steps * vocab];
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < steps; s++) {
                oneHot[(b * steps + s) * vocab + Math.max(0, targets[b][s])] = 1;
            }
        }
        Var logp = Activation.LOG_SOFTMAX.forward(logits);
        return logp.mul(Tensor.of(Shape.of(batch, steps, vocab), oneHot)).sum(2, false);
    }

    /**
     * Summed log-probability of each sequence's tokens from {@code from} on, given the ones
     * before: {@code log p(y | x)} for preference losses.
     *
     * @param sequences {@code [batch][steps]} prompt plus response, the same length
     * @param from the index of the first response token in each row
     * @param mean whether to average over the response tokens instead of summing
     * @return {@code [batch]}
     */
    public Var sequenceLogProb(int[][] sequences, int[] from, boolean mean) {
        int batch = sequences.length;
        int n = sequences[0].length - 1;
        int[][] in = new int[batch][];
        int[][] target = new int[batch][];
        float[] weight = new float[batch * n];
        for (int b = 0; b < batch; b++) {
            in[b] = Arrays.copyOf(sequences[b], n);
            target[b] = Arrays.copyOfRange(sequences[b], 1, n + 1);
            int count = n + 1 - from[b];
            for (int s = 0; s < n; s++) {
                if (s + 1 >= from[b]) {
                    weight[b * n + s] = mean ? 1f / count : 1f;
                }
            }
        }
        Var lp = tokenLogProbs(forward(in).logits(), target);
        return lp.mul(Tensor.of(Shape.of(batch, n), weight)).sum(1, false);
    }

    // ------------------------------------------------------------------ LanguageModel

    /** A state that remembers the tokens and recomputes. */
    private final class TokenState implements State {
        private int[][] tokens;

        TokenState(int batch) {
            tokens = new int[batch][0];
        }

        @Override
        public int length() {
            return tokens[0].length;
        }

        @Override
        public State select(int[] rows) {
            int[][] t = new int[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                t[i] = tokens[rows[i]].clone();
            }
            tokens = t;
            return this;
        }

        @Override
        public State truncate(int length) {
            for (int b = 0; b < tokens.length; b++) {
                tokens[b] = Arrays.copyOf(tokens[b], length);
            }
            return this;
        }

        void append(int[][] more) {
            for (int b = 0; b < tokens.length; b++) {
                int[] t = Arrays.copyOf(tokens[b], tokens[b].length + more[b].length);
                System.arraycopy(more[b], 0, t, tokens[b].length, more[b].length);
                tokens[b] = t;
            }
        }
    }

    @Override
    public State newState(int batch) {
        return new TokenState(batch);
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        if (padding != null) {
            for (boolean[] row : padding) {
                for (boolean p : row) {
                    if (p) {
                        throw new UnsupportedOperationException("HookedDecoder reads no padding");
                    }
                }
            }
        }
        Tensor all = nextAll(state, tokens);
        int batch = tokens.length;
        int steps = all.shape().dim(1);
        return all.slice(1, steps - 1, steps).reshape(batch, settings.vocab());
    }

    @Override
    public Tensor nextAll(State state, int[][] tokens) {
        TokenState s = (TokenState) state;
        s.append(tokens);
        Var logits = forward(s.tokens).logits();
        int steps = logits.shape().dim(1);
        return logits.value().slice(1, steps - tokens[0].length, steps);
    }

    /**
     * Logits of the last position of one sequence, as a float array.
     *
     * @param tokens the sequence
     * @return {@code [vocab]}
     */
    public float[] lastLogits(int[] tokens) {
        Var l = forward(new int[][] {tokens}).logits();
        int steps = l.shape().dim(1);
        return l.value().slice(1, steps - 1, steps).toArray();
    }

    @Override
    public String toString() {
        return "HookedDecoder(" + settings + ")";
    }
}
