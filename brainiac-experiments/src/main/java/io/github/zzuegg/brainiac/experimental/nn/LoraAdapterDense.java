package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Quantization;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A low-rank adapter around a frozen core {@link Dense}: {@code y = act(x W0 + b + s x A B)}
 * with {@code A [in, r]}, {@code B [r, out]} and {@code s = alpha / r}. Five starts:
 * <ul>
 * <li>{@link #lora}: LoRA (Hu et al. 2021, <a href="https://arxiv.org/abs/2106.09685">arXiv:2106.09685</a>),
 *     {@code A} random and {@code B} zero, so training starts at the original layer.</li>
 * <li>{@link #pissa}: PiSSA (Meng et al. 2024, <a href="https://arxiv.org/abs/2404.02948">arXiv:2404.02948</a>),
 *     {@code A = U_r S_r^1/2}, {@code B = S_r^1/2 V_r^T} from the weight's top singular
 *     directions, with the frozen part the residual {@code W0 - A B}: the same function at
 *     the start, but the adapter trains the principal components, which converges faster
 *     than LoRA's zero start.</li>
 * <li>{@link #rsLora}: rank-stabilized scaling {@code alpha / sqrt(r)}.</li>
 * <li>{@link #qlora} and {@link #loftQ}: adapters over a quantized frozen weight, started at
 *     zero (QLoRA) or at the low-rank part of the quantization error (LoftQ).</li>
 * </ul>
 * Pair it with {@code LoraPlus} for LoRA+'s larger learning rate on {@code B}. The core
 * {@code Llama.addLora} adapts Llama's own projections; this one adapts any {@code Dense}.
 */
public final class LoraAdapterDense implements Layer {

    private final Dense base;
    private final Tensor frozen;
    private final Tensor bias;
    private final Var a;
    private final Var b;
    private final float scale;

    private LoraAdapterDense(Dense base, Tensor frozen, Tensor a, Tensor b, float scale) {
        this.base = base;
        this.frozen = frozen;
        this.bias = base.bias().value();
        this.a = Var.param(a);
        this.b = Var.param(b);
        this.scale = scale;
    }

    /**
     * @param base the frozen layer
     * @param rank the rank
     * @param alpha the scale numerator ({@code s = alpha / rank})
     * @param random the source of {@code A}
     * @return a LoRA adapter
     */
    public static LoraAdapterDense lora(Dense base, int rank, float alpha, RandomGenerator random) {
        Tensor w = base.weight().value();
        int in = w.shape().dim(0);
        int out = w.shape().dim(1);
        Tensor a = Tensor.normal(Shape.of(in, rank), 0, (float) (1 / Math.sqrt(in)), random);
        return new LoraAdapterDense(base, w, a, Tensor.zeros(Shape.of(rank, out)), alpha / rank);
    }

    /**
     * @param base the layer whose principal directions become the adapter
     * @param rank the rank
     * @return a PiSSA adapter (scale 1)
     */
    public static LoraAdapterDense pissa(Dense base, int rank) {
        Tensor w = base.weight().value();
        ThinSvd svd = ThinSvd.of(w);
        Tensor a = svd.left(rank, 0.5);
        Tensor b = svd.right(rank, 0.5);
        Tensor residual = w.sub(a.matmul(b));
        return new LoraAdapterDense(base, residual, a, b, 1f);
    }

    /**
     * Rank-stabilized LoRA (rsLoRA, Kalajdzievski 2023,
     * <a href="https://arxiv.org/abs/2312.03732">arXiv:2312.03732</a>): LoRA with the scale
     * {@code alpha / sqrt(r)} instead of {@code alpha / r}. With {@code alpha / r} the
     * adapter's output and gradients shrink as the rank grows, so higher ranks learn slower
     * and stop paying off; {@code 1 / sqrt(r)} is the scaling that keeps them stable, so loss
     * keeps improving with rank.
     *
     * @param base the frozen layer
     * @param rank the rank
     * @param alpha the scale numerator ({@code s = alpha / sqrt(rank)})
     * @param random the source of {@code A}
     * @return an rsLoRA adapter
     */
    public static LoraAdapterDense rsLora(Dense base, int rank, float alpha, RandomGenerator random) {
        Tensor w = base.weight().value();
        int in = w.shape().dim(0);
        int out = w.shape().dim(1);
        Tensor a = Tensor.normal(Shape.of(in, rank), 0, (float) (1 / Math.sqrt(in)), random);
        return new LoraAdapterDense(base, w, a, Tensor.zeros(Shape.of(rank, out)), (float) (alpha / Math.sqrt(rank)));
    }

    /**
     * QLoRA's starting point (Dettmers et al. 2023,
     * <a href="https://arxiv.org/abs/2305.14314">arXiv:2305.14314</a>): the frozen weight is
     * replaced by its {@code bits}-bit quantization (symmetric, per output channel, core
     * {@link Quantization}) and an ordinary LoRA adapter is trained on top. The adapter starts
     * at zero, so the model starts with the full quantization error.
     *
     * @param base the layer to quantize and adapt
     * @param rank the rank
     * @param alpha the scale numerator ({@code s = alpha / rank})
     * @param bits the bits of the frozen weight, for example 4
     * @param random the source of {@code A}
     * @return the adapter over a quantized base
     */
    public static LoraAdapterDense qlora(Dense base, int rank, float alpha, int bits, RandomGenerator random) {
        Tensor w = base.weight().value();
        int in = w.shape().dim(0);
        int out = w.shape().dim(1);
        Tensor q = Quantization.bits(bits).quantize(w, 1);
        Tensor a = Tensor.normal(Shape.of(in, rank), 0, (float) (1 / Math.sqrt(in)), random);
        return new LoraAdapterDense(base, q, a, Tensor.zeros(Shape.of(rank, out)), alpha / rank);
    }

    /**
     * LoftQ (Li et al. 2023, <a href="https://arxiv.org/abs/2310.08659">arXiv:2310.08659</a>):
     * quantization-aware LoRA initialization. Alternates {@code Q = quant(W - A B)} and
     * {@code A B = SVD_r(W - Q)}, so that {@code Q + A B} approximates the full-precision
     * weight from the start; the adapter begins by undoing the quantization error that QLoRA
     * would leave the fine-tuning to recover.
     *
     * @param base the layer to quantize and adapt
     * @param rank the rank
     * @param bits the bits of the frozen weight, for example 4
     * @param iterations alternating steps, for example 5
     * @return the adapter (scale 1) over the quantized residual
     */
    public static LoraAdapterDense loftQ(Dense base, int rank, int bits, int iterations) {
        Tensor w = base.weight().value();
        Quantization quant = Quantization.bits(bits);
        Tensor a = Tensor.zeros(Shape.of(w.shape().dim(0), rank));
        Tensor b = Tensor.zeros(Shape.of(rank, w.shape().dim(1)));
        Tensor q = quant.quantize(w, 1);
        for (int it = 0; it < Math.max(1, iterations); it++) {
            q = quant.quantize(w.sub(a.matmul(b)), 1);
            ThinSvd svd = ThinSvd.of(w.sub(q));
            a = svd.left(rank, 0.5);
            b = svd.right(rank, 0.5);
        }
        return new LoraAdapterDense(base, q, a, b, 1f);
    }

    /** @return the frozen weight the adapter is added to (quantized for QLoRA and LoftQ) */
    public Tensor frozen() {
        return frozen;
    }

    /** @return {@code A [in, r]} */
    public Var a() {
        return a;
    }

    /** @return {@code B [r, out]} */
    public Var b() {
        return b;
    }

    /** @return the effective weight {@code W0 + s A B} */
    public Var weight() {
        return a.matmul(b).mul(scale).add(frozen);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var y = input.matmul(Var.of(frozen)).add(bias).add(input.matmul(a).matmul(b).mul(scale));
        return base.activation().forward(y, training);
    }

    @Override
    public List<Var> parameters() {
        return List.of(a, b);
    }

    @Override
    public int outputSize(int inputSize) {
        return base.outputSize(inputSize);
    }

    /** @return a plain {@code Dense} with the adapter folded in */
    public Dense merge() {
        Tensor w = weight().value();
        return new Dense(w.shape().dim(0), w.shape().dim(1), base.activation(), w, bias);
    }

    @Override
    public String toString() {
        return "LoraAdapterDense(" + base + ", rank " + a.shape().dim(1) + ")";
    }
}
