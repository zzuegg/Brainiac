package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Slot attention {@code [batch, n, inputs] -> [batch, slots, width]}: {@code K} slots, drawn
 * from a learned Gaussian, iteratively compete for the input features. Each of {@code T}
 * iterations computes attention logits between the inputs (keys) and the normalised slots
 * (queries), takes the softmax <em>over the slots</em> (so slots compete to explain each
 * input), renormalises over the inputs into a weighted mean of the values, and updates each
 * slot with a shared {@link GruCell} and a residual MLP. Trained through a reconstruction
 * loss, the slots bind to objects without supervision; {@link #lastAttention()} is the soft
 * segmentation.
 *
 * <p>The inverted softmax is what separates it from the plain cross-attention of
 * {@link LatentCrossAttention} (where queries do not compete).
 *
 * <p>Paper: Locatello et al., "Object-Centric Learning with Slot Attention" (NeurIPS 2020),
 * <a href="https://arxiv.org/abs/2006.15055">arXiv:2006.15055</a>.
 */
public final class SlotAttention implements Layer {

    private final int inputs;
    private final int slots;
    private final int width;
    private final int iterations;
    private final LayerNorm inputNorm;
    private final LayerNorm slotNorm;
    private final LayerNorm mlpNorm;
    private final Var key;
    private final Var value;
    private final Var query;
    private final Var mu;
    private final Var logSigma;
    private final GruCell gru;
    private final Var mlpUp;
    private final Var mlpUpBias;
    private final Var mlpDown;
    private final Var mlpDownBias;
    private final RandomGenerator random;
    private Tensor lastAttention;

    /**
     * Creates the module.
     *
     * @param inputs the features per input element
     * @param slots the number of slots {@code K}
     * @param width the slot width
     * @param iterations the attention iterations {@code T} (3 in the paper)
     * @param random the source of weights and slot noise
     */
    public SlotAttention(int inputs, int slots, int width, int iterations, RandomGenerator random) {
        this.inputs = inputs;
        this.slots = slots;
        this.width = width;
        this.iterations = iterations;
        this.inputNorm = LayerNorm.of(inputs);
        this.slotNorm = LayerNorm.of(width);
        this.mlpNorm = LayerNorm.of(width);
        this.key = ClassicOps.matrix(random, inputs, width);
        this.value = ClassicOps.matrix(random, inputs, width);
        this.query = ClassicOps.matrix(random, width, width);
        this.mu = Var.param(Tensor.normal(Shape.of(1, 1, width), 0, 0.5f, random));
        this.logSigma = Var.param(Tensor.full(Shape.of(1, 1, width), -1f));
        this.gru = new GruCell(width, width, random);
        this.mlpUp = ClassicOps.matrix(random, width, 2 * width);
        this.mlpUpBias = ClassicOps.zeros(2 * width);
        this.mlpDown = ClassicOps.matrix(random, 2 * width, width);
        this.mlpDownBias = ClassicOps.zeros(width);
        this.random = random;
    }

    /**
     * Describes the module for model builders on {@code [n, inputs]}.
     *
     * @param slots the number of slots
     * @param width the slot width
     * @param iterations the iterations
     * @return the spec
     */
    public static LayerSpec spec(int slots, int width, int iterations) {
        return ClassicOps.spec(s -> Shape.of(slots, width),
                (s, r) -> new SlotAttention(s.dim(1), slots, width, iterations, r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return forward(input, null, training);
    }

    /**
     * Slot attention starting from given slots instead of a draw from the learned Gaussian, as
     * video models (SAVi, SlotFormer) initialise each frame's slots with the previous frame's,
     * so that slot {@code k} keeps following the same object.
     *
     * @param input {@code [batch, n, inputs]}
     * @param initial the starting slots {@code [batch, slots, width]}, or {@code null} to draw them
     * @param training whether training (unused; kept for symmetry with {@link #forward(Var, boolean)})
     * @return the slots {@code [batch, slots, width]}
     */
    public Var forward(Var input, Var initial, boolean training) {
        if (input.shape().rank() != 3 || input.shape().dim(2) != inputs) {
            throw new IllegalArgumentException(this + " expects [batch, n, " + inputs + "], got " + input.shape());
        }
        int b = input.shape().dim(0);
        int n = input.shape().dim(1);
        Var x = inputNorm.forward(input);
        Var k = x.matmul(key);
        Var v = x.matmul(value);
        Var s;
        if (initial == null) {
            Tensor noise = Tensor.normal(Shape.of(b, slots, width), 0, 1, random);
            s = mu.add(logSigma.exp().mul(noise));
        } else {
            s = initial;
        }
        float scale = (float) (1 / Math.sqrt(width));
        Var attn = null;
        for (int t = 0; t < iterations; t++) {
            Var q = slotNorm.forward(s).matmul(query);
            Var logits = k.matmul(q.transpose()).mul(scale);                 // [b, n, K]
            attn = Activation.SOFTMAX.forward(logits).add(1e-8f);            // softmax over slots
            Var w = attn.div(attn.sum(1, true));                            // weighted mean over inputs
            Var updates = w.transpose().matmul(v);                          // [b, K, width]
            Var next = gru.step(updates.reshape(b * slots, width), s.reshape(b * slots, width))
                    .reshape(b, slots, width);
            Var h = mlpNorm.forward(next);
            s = next.add(Activation.RELU.forward(h.matmul(mlpUp).add(mlpUpBias)).matmul(mlpDown).add(mlpDownBias));
        }
        lastAttention = attn.value();
        if (n == 0) {
            throw new IllegalArgumentException("no inputs");
        }
        return s;
    }

    /** @return the soft assignment {@code [batch, n, slots]} of the last forward pass */
    public Tensor lastAttention() {
        return lastAttention;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(slots, width);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new java.util.ArrayList<>(inputNorm.parameters());
        p.addAll(slotNorm.parameters());
        p.addAll(mlpNorm.parameters());
        p.addAll(List.of(key, value, query, mu, logSigma));
        p.addAll(gru.parameters());
        p.addAll(List.of(mlpUp, mlpUpBias, mlpDown, mlpDownBias));
        return p;
    }

    @Override
    public String toString() {
        return "SlotAttention(" + slots + " slots x " + width + ", " + iterations + " iterations)";
    }
}
