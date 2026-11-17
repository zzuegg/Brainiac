package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.llm.Rope;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A core {@link Llama}'s weights run over a pluggable {@link KvBlockStore}, for serving: the
 * sequences of one step may have different lengths and different numbers of new tokens (a
 * ragged batch), so a scheduler can mix prefill chunks of new prompts with single decode
 * tokens of running ones in one pass, and every projection runs once over all of them. Each
 * sequence attends only to its own slots, through its own block table.
 *
 * <p>A {@link Pass} can also carry a token tree: each new token names its parent among the
 * pass's tokens and attends to the cached prefix plus its ancestors only (tree attention, as
 * EAGLE and Medusa verify drafts), at position {@code length + depth}.
 *
 * <p>It is also a {@link LanguageModel}, so the core {@code Generator} (sampling, speculative
 * decoding, beam search with copy-on-write forks) and {@link PrefixCache} drive it unchanged;
 * its logits equal {@link Llama#next}'s. The core {@code Llama} keeps its contiguous cache
 * private, so this class re-runs the decoder over {@link Llama#namedParameters()}: weights
 * and biases as stored (float32 compute), without LoRA adapters or fused projections. The
 * core keeps {@code Llama.prefillChunk} and {@code Llama.slidingWindow} for its own cache;
 * here chunking is the {@linkplain ChunkedPrefillScheduler scheduler's} job.
 */
public final class PagedLlama implements LanguageModel {

    /**
     * New tokens for one sequence.
     *
     * @param sequence the store's sequence id
     * @param tokens the new tokens
     * @param parents per token, the index of its parent among {@code tokens}, or -1 when it
     *     follows the cached prefix directly; must be smaller than the token's own index
     */
    public record Pass(int sequence, int[] tokens, int[] parents) {

        /** Checks the tree. */
        public Pass {
            if (tokens.length == 0 || parents.length != tokens.length) {
                throw new IllegalArgumentException("need at least one token and one parent per token");
            }
            for (int i = 0; i < parents.length; i++) {
                if (parents[i] < -1 || parents[i] >= i) {
                    throw new IllegalArgumentException("parent of token " + i + " must be -1 or below " + i);
                }
            }
        }

        /**
         * A plain run of tokens, each following the one before.
         *
         * @param sequence the sequence id
         * @param tokens the tokens
         * @return the pass
         */
        public static Pass chain(int sequence, int... tokens) {
            int[] p = new int[tokens.length];
            for (int i = 0; i < p.length; i++) {
                p[i] = i - 1;
            }
            return new Pass(sequence, tokens.clone(), p);
        }
    }

    /**
     * What one pass gives back.
     *
     * @param logits float32 logits {@code [steps * vocab]}, row by row
     * @param hidden the final normalized hidden states {@code [steps * hidden]}
     * @param steps the rows: every new token, or only the last
     */
    public record Output(float[] logits, float[] hidden, int steps) {

        /**
         * @param step a row
         * @param vocab the vocabulary size
         * @return that row's logits
         */
        public float[] row(int step, int vocab) {
            return Arrays.copyOfRange(logits, step * vocab, (step + 1) * vocab);
        }
    }

    private final Llama model;
    private final KvBlockStore store;
    private final LlamaConfig config;
    private final Rope rope;
    private final Device device;
    private long passes;
    private long tokensRead;

    /**
     * @param model the weights; any type, computed in float32
     * @param store where keys and values go, shaped for the model
     */
    public PagedLlama(Llama model, KvBlockStore store) {
        this.model = Objects.requireNonNull(model, "model");
        this.store = Objects.requireNonNull(store, "store");
        this.config = model.config();
        if (store.layers() != config.layers()) {
            throw new IllegalArgumentException("the store has " + store.layers() + " layers, the model "
                    + config.layers());
        }
        this.rope = new Rope(config.rope());
        this.device = model.device();
    }

    /**
     * A model over a fresh {@link PagedKvCache} on the model's device.
     *
     * @param model the weights
     * @param blockSize the slots per block
     * @param blocks the blocks in the pool
     * @return the paged model
     */
    public static PagedLlama of(Llama model, int blockSize, int blocks) {
        return new PagedLlama(model, PagedKvCache.forModel(model.config(), blockSize, blocks, model.device()));
    }

    /** @return the store */
    public KvBlockStore store() {
        return store;
    }

    /** @return the weights */
    public Llama model() {
        return model;
    }

    /** @return the model passes run so far */
    public long passes() {
        return passes;
    }

    /** @return the new tokens read so far, over all passes */
    public long tokensRead() {
        return tokensRead;
    }

    @Override
    public int vocabSize() {
        return config.vocabSize();
    }

    /** @return the hidden size */
    public int hiddenSize() {
        return config.hidden();
    }

    /**
     * The output layer on hidden states, as the model applies it: for draft heads that predict
     * hidden states (EAGLE).
     *
     * @param hidden {@code [rows * hidden]}
     * @return logits {@code [rows * vocab]}
     */
    public float[] project(float[] hidden) {
        int rows = hidden.length / config.hidden();
        return Var.of(Tensor.of(Shape.of(rows, config.hidden()), hidden, device)).matmul(head()).value()
                .to(DType.FLOAT32).toArray();
    }

    /**
     * Embeds tokens with the model's own table.
     *
     * @param tokens the tokens
     * @return {@code [tokens * hidden]}
     */
    public float[] embed(int... tokens) {
        return weight("model.embed_tokens.weight").gather(0, indices(tokens)).value().toArray();
    }

    /**
     * Runs one step over several sequences at once.
     *
     * @param passes the new tokens of each sequence; each sequence at most once
     * @param everyStep whether to return every new token's row or only the last
     * @return one output per pass
     * @throws IllegalStateException if the store cannot hold the new tokens (nothing changes)
     */
    public List<Output> forward(List<Pass> passes, boolean everyStep) {
        int count = passes.size();
        int total = 0;
        for (Pass p : passes) {
            if (!store.canAppend(p.sequence(), p.tokens().length)) {
                throw new IllegalStateException("the store has no room for " + p.tokens().length
                        + " more tokens of sequence " + p.sequence());
            }
            total += p.tokens().length;
        }
        int[] all = new int[total];
        int[] offsets = new int[count];
        int[] starts = new int[count];
        int[][] positions = new int[count][];
        boolean[][][] visible = new boolean[count][][];
        int o = 0;
        for (int i = 0; i < count; i++) {
            Pass p = passes.get(i);
            int n = p.tokens().length;
            offsets[i] = o;
            System.arraycopy(p.tokens(), 0, all, o, n);
            o += n;
            int length = store.length(p.sequence());
            positions[i] = new int[n];
            visible[i] = new boolean[n][n];
            for (int t = 0; t < n; t++) {
                int parent = p.parents()[t];
                positions[i][t] = parent < 0 ? length : positions[i][parent] + 1;
                if (parent >= 0) {
                    System.arraycopy(visible[i][parent], 0, visible[i][t], 0, n);
                }
                visible[i][t][t] = true;
            }
            starts[i] = store.append(p.sequence(), n);
        }
        int hd = config.headDim();
        int heads = config.heads();
        int kvHeads = config.kvHeads();
        float scale = (float) (1 / Math.sqrt(hd));
        Var x = weight("model.embed_tokens.weight").gather(0, indices(all));
        for (int l = 0; l < config.layers(); l++) {
            String pre = "model.layers." + l + ".";
            Var h = rmsNorm(x, weight(pre + "input_layernorm.weight"));
            Var q = linear(h, pre + "self_attn.q_proj");
            Var k = linear(h, pre + "self_attn.k_proj");
            Var v = linear(h, pre + "self_attn.v_proj");
            Var[] parts = new Var[count];
            for (int i = 0; i < count; i++) {
                Pass p = passes.get(i);
                int n = p.tokens().length;
                int from = offsets[i];
                int[][] pos = {positions[i]};
                Var qp = rope.apply(q.slice(0, from, from + n).reshape(1, n, heads, hd).permute(0, 2, 1, 3), pos);
                Var kp = rope.apply(k.slice(0, from, from + n).reshape(1, n, kvHeads, hd).permute(0, 2, 1, 3), pos);
                Tensor kRows = kp.permute(0, 2, 1, 3).reshape(n, kvHeads, hd).value();
                Tensor vRows = v.slice(0, from, from + n).reshape(n, kvHeads, hd).value();
                store.write(p.sequence(), l, starts[i], kRows, vRows);
                Tensor queries = qp.value().reshape(heads, n, hd);
                KvBlockStore.View view = store.view(p.sequence(), l, queries);
                int m = view.slots().length;
                float[] mask = new float[n * m];
                for (int t = 0; t < n; t++) {
                    for (int j = 0; j < m; j++) {
                        int rel = view.slots()[j] - starts[i];
                        if (rel >= 0 && (rel >= n || !visible[i][t][rel])) {
                            mask[t * m + j] = -1e9f;
                        }
                    }
                }
                Var keys = Var.of(view.keys()).reshape(1, m, kvHeads, hd).permute(0, 2, 1, 3);
                Var values = Var.of(view.values()).reshape(1, m, kvHeads, hd).permute(0, 2, 1, 3);
                Tensor bias = Tensor.of(Shape.of(1, 1, n, m), mask, device);
                Var a = Functional.attention(Var.of(qp.value()), keys, values, bias, scale, 0, null);
                parts[i] = a.permute(0, 2, 1, 3).reshape(n, heads * hd);
            }
            Var attn = count == 1 ? parts[0] : Var.concat(0, parts);
            x = x.add(linear(attn, pre + "self_attn.o_proj"));
            h = rmsNorm(x, weight(pre + "post_attention_layernorm.weight"));
            Var gated = linear(h, pre + "mlp.gate_proj").activation(ActivationOp.SILU, 0)
                    .mul(linear(h, pre + "mlp.up_proj"));
            x = x.add(linear(gated, pre + "mlp.down_proj"));
        }
        if (!everyStep) {
            Var[] last = new Var[count];
            for (int i = 0; i < count; i++) {
                int end = offsets[i] + passes.get(i).tokens().length;
                last[i] = x.slice(0, end - 1, end);
            }
            x = count == 1 ? last[0] : Var.concat(0, last);
        }
        Var hidden = rmsNorm(x, weight("model.norm.weight"));
        float[] logits = hidden.matmul(head()).value().to(DType.FLOAT32).toArray();
        float[] states = hidden.value().to(DType.FLOAT32).toArray();
        this.passes++;
        this.tokensRead += total;
        int vocab = config.vocabSize();
        int dim = config.hidden();
        List<Output> out = new ArrayList<>(count);
        int row = 0;
        for (Pass p : passes) {
            int rows = everyStep ? p.tokens().length : 1;
            out.add(new Output(Arrays.copyOfRange(logits, row * vocab, (row + rows) * vocab),
                    Arrays.copyOfRange(states, row * dim, (row + rows) * dim), rows));
            row += rows;
        }
        return out;
    }

    // ---------------------------------------------------------------- LanguageModel

    /** One sequence per row, with which steps were padding. */
    private final class Rows implements State {
        final int[] sequences;
        boolean[][] padding;
        float[][] last;

        Rows(int[] sequences, boolean[][] padding, float[][] last) {
            this.sequences = sequences;
            this.padding = padding;
            this.last = last;
        }

        @Override
        public int length() {
            return padding.length == 0 ? 0 : padding[0].length;
        }

        @Override
        public State select(int[] rows) {
            int[] seqs = new int[rows.length];
            boolean[][] pad = new boolean[rows.length][];
            float[][] l = new float[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                seqs[i] = store.fork(sequences[rows[i]]);
                pad[i] = padding[rows[i]].clone();
                l[i] = last[rows[i]];
            }
            for (int s : sequences) {
                store.free(s);
            }
            return new Rows(seqs, pad, l);
        }

        @Override
        public State truncate(int length) {
            for (int b = 0; b < sequences.length; b++) {
                int real = 0;
                for (int t = 0; t < length; t++) {
                    real += padding[b][t] ? 0 : 1;
                }
                store.truncate(sequences[b], real);
                padding[b] = Arrays.copyOf(padding[b], length);
            }
            return this;
        }
    }

    @Override
    public State newState(int batch) {
        int[] seqs = new int[batch];
        for (int b = 0; b < batch; b++) {
            seqs[b] = store.newSequence();
        }
        return new Rows(seqs, new boolean[batch][0], new float[batch][]);
    }

    /**
     * Gives a state's memory back to the store. The core {@code LanguageModel.State} has no
     * such hook, so a paged pool shared by many requests needs this call.
     *
     * @param state a state of this model
     */
    public void release(State state) {
        for (int s : ((Rows) state).sequences) {
            store.free(s);
        }
    }

    /**
     * An independent copy of a state that shares its blocks until either writes (copy on
     * write); the original stays usable. The core {@code LanguageModel.State} has no copy, so
     * {@link PrefixCache} takes this as a function.
     *
     * @param state a state of this model
     * @return the copy
     */
    public State fork(State state) {
        Rows r = (Rows) state;
        int[] seqs = new int[r.sequences.length];
        boolean[][] pad = new boolean[seqs.length][];
        for (int b = 0; b < seqs.length; b++) {
            seqs[b] = store.fork(r.sequences[b]);
            pad[b] = r.padding[b].clone();
        }
        return new Rows(seqs, pad, r.last.clone());
    }

    /**
     * @param state a state of this model
     * @param row a row
     * @return the store's sequence id behind that row
     */
    public int sequence(State state, int row) {
        return ((Rows) state).sequences[row];
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        return read((Rows) state, tokens, padding, false);
    }

    @Override
    public Tensor nextAll(State state, int[][] tokens) {
        return read((Rows) state, tokens, null, true);
    }

    private Tensor read(Rows rows, int[][] tokens, boolean[][] padding, boolean everyStep) {
        int batch = tokens.length;
        if (batch != rows.sequences.length) {
            throw new IllegalArgumentException("the state holds " + rows.sequences.length + " sequences, got " + batch);
        }
        int steps = tokens[0].length;
        List<Pass> passes = new ArrayList<>();
        List<Integer> owners = new ArrayList<>();
        for (int b = 0; b < batch; b++) {
            int[] real = new int[steps];
            int n = 0;
            for (int t = 0; t < steps; t++) {
                if (padding == null || !padding[b][t]) {
                    real[n++] = tokens[b][t];
                }
            }
            if (everyStep && n != steps) {
                throw new IllegalArgumentException("every step needs every token");
            }
            if (n > 0) {
                passes.add(Pass.chain(rows.sequences[b], Arrays.copyOf(real, n)));
                owners.add(b);
            }
            int old = rows.padding[b].length;
            rows.padding[b] = Arrays.copyOf(rows.padding[b], old + steps);
            for (int t = 0; t < steps; t++) {
                rows.padding[b][old + t] = padding != null && padding[b][t];
            }
        }
        int vocab = config.vocabSize();
        List<Output> outs = passes.isEmpty() ? List.of() : forward(passes, everyStep);
        for (int i = 0; i < outs.size(); i++) {
            Output out = outs.get(i);
            rows.last[owners.get(i)] = out.row(out.steps() - 1, vocab);
        }
        if (everyStep) {
            float[] flat = new float[batch * steps * vocab];
            for (int i = 0; i < outs.size(); i++) {
                System.arraycopy(outs.get(i).logits(), 0, flat, owners.get(i) * steps * vocab, steps * vocab);
            }
            return Tensor.of(Shape.of(batch, steps, vocab), flat);
        }
        float[] flat = new float[batch * vocab];
        for (int b = 0; b < batch; b++) {
            if (rows.last[b] != null) {
                System.arraycopy(rows.last[b], 0, flat, b * vocab, vocab);
            }
        }
        return Tensor.of(Shape.of(batch, vocab), flat);
    }

    // ---------------------------------------------------------------- weights

    private Var weight(String name) {
        Map<String, Var> named = model.namedParameters();
        Var w = named.get(name);
        if (w == null) {
            throw new IllegalStateException("the model has no " + name);
        }
        Tensor t = w.value();
        return Var.of(t.dtype() == DType.FLOAT32 ? t : t.to(DType.FLOAT32));
    }

    private Var linear(Var x, String module) {
        Var y = x.matmul(weight(module + ".weight"));
        Var b = model.namedParameters().get(module + ".bias");
        if (b != null) {
            y = y.add(Var.of(b.value().to(DType.FLOAT32)));
        }
        return y;
    }

    private Var head() {
        return config.tieEmbeddings() ? weight("model.embed_tokens.weight").transpose() : weight("lm_head.weight");
    }

    private Var rmsNorm(Var x, Var weight) {
        int last = x.shape().rank() - 1;
        return x.div(x.square().mean(last, true).add(config.rmsEps()).sqrt()).mul(weight);
    }

    private Tensor indices(int[] ids) {
        float[] f = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            f[i] = ids[i];
        }
        return Tensor.of(Shape.of(ids.length), f, device);
    }

    @Override
    public String toString() {
        return "PagedLlama[" + model + ", " + store + "]";
    }
}
