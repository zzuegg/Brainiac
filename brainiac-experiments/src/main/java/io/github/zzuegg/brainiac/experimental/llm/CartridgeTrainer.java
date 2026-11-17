package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Cartridges (Eyuboglu et al., "Cartridges: lightweight and general-purpose long context
 * representations via self-study", <a href="https://arxiv.org/abs/2506.06266">arXiv:2506.06266</a>):
 * instead of keeping a long corpus in the prompt, and its KV cache in memory, for every
 * question, a small trainable KV cache, the cartridge, is trained offline for that one corpus and
 * then loaded in its place. It is trained by <em>context distillation</em> on synthetic
 * "self-study" queries about the corpus: the teacher is the same frozen model with the whole
 * corpus in context, the student reads the cartridge instead, and the student's next-token
 * distributions are matched to the teacher's (KL), which the paper found to generalise far
 * better than next-token training on the synthetic text.
 *
 * <p>The cartridge is a per-layer key and value prefix of a {@link HookedDecoder} (the
 * mechanism of {@link SoftPrompts#prefixTuning}), started, as in the paper, from the keys and
 * values of the first corpus tokens. The model's own weights stay frozen.
 */
public final class CartridgeTrainer {
    private final HookedDecoder teacher;
    private final HookedDecoder student;
    private final int[] corpus;
    private final int tokens;
    private final List<Var> parameters = new ArrayList<>();

    private CartridgeTrainer(HookedDecoder teacher, int[] corpus, int tokens) {
        this.teacher = teacher;
        this.student = teacher.copy();
        this.corpus = corpus.clone();
        this.tokens = tokens;
    }

    /**
     * Creates a cartridge for one corpus.
     *
     * @param model the frozen model; it is the teacher and is never changed
     * @param corpus the corpus the cartridge stands for
     * @param tokens the cartridge length, in KV entries per layer
     * @return the trainer
     */
    public static CartridgeTrainer of(HookedDecoder model, int[] corpus, int tokens) {
        if (tokens < 1) {
            throw new IllegalArgumentException("a cartridge needs at least one token");
        }
        CartridgeTrainer t = new CartridgeTrainer(Objects.requireNonNull(model), corpus, tokens);
        t.initialize();
        return t;
    }

    private void initialize() {
        float[] ids = new float[tokens];
        for (int i = 0; i < tokens; i++) {
            ids[i] = corpus[i % corpus.length];
        }
        Tensor e = student.embedding().value().gather(0, Tensor.of(Shape.of(tokens), ids));
        for (int i = 0; i < student.layers(); i++) {
            Layer k = student.linears().get("block." + i + ".attn.k");
            Layer v = student.linears().get("block." + i + ".attn.v");
            Var pk = Var.param(k.forward(Var.of(e)).value());
            Var pv = Var.param(v.forward(Var.of(e)).value());
            student.prefix(i, pk, pv);
            parameters.add(pk);
            parameters.add(pv);
        }
    }

    /**
     * One context-distillation step on a batch of self-study queries of equal length.
     *
     * @param queries synthetic queries about the corpus
     * @param optimizer the cartridge's optimizer
     * @return the mean KL per position from the teacher to the student
     */
    public float step(List<int[]> queries, Optimizer optimizer) {
        int batch = queries.size();
        int steps = queries.get(0).length;
        int[][] withCorpus = new int[batch][];
        int[][] alone = queries.toArray(int[][]::new);
        for (int i = 0; i < batch; i++) {
            withCorpus[i] = Arrays.copyOf(corpus, corpus.length + steps);
            System.arraycopy(alone[i], 0, withCorpus[i], corpus.length, steps);
        }
        Var t = teacher.forward(withCorpus).logits();
        int total = t.shape().dim(1);
        Tensor target = Activation.SOFTMAX.forward(Var.of(t.value().slice(1, total - steps, total))).value();
        Var logq = Activation.LOG_SOFTMAX.forward(student.forward(alone).logits());
        Var logp = Var.of(target.add(1e-9f).log());
        Var kl = Var.of(target).mul(logp.sub(logq)).sum(2, false).mean();
        float value = kl.value().item();
        kl.backward();
        optimizer.step(parameters);
        student.zeroGrad();
        teacher.zeroGrad();
        return value;
    }

    /** @return the model reading the cartridge; ask it questions without the corpus */
    public HookedDecoder student() {
        return student;
    }

    /** @return the trainable keys and values, two per layer */
    public List<Var> parameters() {
        return List.copyOf(parameters);
    }

    /**
     * The memory of {@code entries} KV entries of a decoder at 32-bit floats.
     *
     * @param model the decoder
     * @param entries cached positions
     * @return bytes
     */
    public static long kvBytes(HookedDecoder model, int entries) {
        return 2L * model.layers() * model.settings().dim() * entries * Float.BYTES;
    }

    /** @return this cartridge's memory */
    public long kvBytes() {
        return kvBytes(student, tokens);
    }

    @Override
    public String toString() {
        return "CartridgeTrainer(tokens=" + tokens + ", corpus=" + corpus.length + ")";
    }
}
