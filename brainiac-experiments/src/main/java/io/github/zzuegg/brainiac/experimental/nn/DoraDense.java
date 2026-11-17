package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * DoRA, weight-decomposed low-rank adaptation (Liu et al. 2024,
 * <a href="https://arxiv.org/abs/2402.09353">arXiv:2402.09353</a>), for a frozen core
 * {@link Dense} layer. The fine-tuned weight is split into a per-unit <em>magnitude</em>
 * {@code m} and a <em>direction</em>: {@code W' = m * (W0 + A B) / |W0 + A B|_col}, with
 * the column norm taken per output unit. Only {@code m} ({@code [1, out]}) and the low-rank
 * pair {@code A} ({@code [in, r]}, random) and {@code B} ({@code [r, out]}, zero) train, so
 * the layer starts exactly equal to the original. Learning magnitude and direction
 * separately behaves more like full fine-tuning than LoRA does, at the same cost.
 *
 * <p>The low-rank pair is self-contained here (plain LoRA lives in the core).
 * {@link #adapt(Model, int, long)} swaps every {@code Dense} of a sequential model, and
 * {@link #merge()} folds the adapter back into an ordinary {@code Dense}.
 */
public final class DoraDense implements Layer {

    private final Dense base;
    private final Var magnitude;
    private final Var a;
    private final Var b;
    private final Tensor bias;

    /**
     * @param base the frozen layer
     * @param rank the adapter rank
     * @param random the source of {@code A}'s initial values
     */
    public DoraDense(Dense base, int rank, SplittableRandom random) {
        this.base = base;
        Tensor w = base.weight().value();
        int in = w.shape().dim(0);
        int out = w.shape().dim(1);
        this.magnitude = Var.param(w.square().sum(0, true).sqrt());
        this.a = Var.param(Tensor.normal(Shape.of(in, rank), 0, (float) Math.sqrt(1.0 / in), random));
        this.b = Var.param(Tensor.zeros(Shape.of(rank, out)));
        this.bias = base.bias().value();
    }

    /**
     * A copy of a sequential model with every {@link Dense} layer replaced by a DoRA
     * adapter around it (the original weights are shared, not copied, and stay frozen).
     *
     * @param model a sequential model
     * @param rank the adapter rank
     * @param seed the seed of the adapters
     * @return the adapted model; its {@code parameters()} are only the adapters'
     */
    public static Model adapt(Model model, int rank, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Model.Builder builder = Model.sequential().input(model.inputShape().toArray());
        for (Layer l : model.layers()) {
            builder.layer(l instanceof Dense d ? new DoraDense(d, rank, r) : l);
        }
        return builder.build();
    }

    /** @return the combined weight {@code m * V / |V|_col} */
    public Var weight() {
        Var v = a.matmul(b).add(base.weight().value());
        Var norm = v.square().sum(0, true).sqrt();
        return v.div(norm).mul(magnitude);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Activation act = base.activation();
        return act.forward(input.matmul(weight()).add(bias), training);
    }

    @Override
    public List<Var> parameters() {
        return List.of(magnitude, a, b);
    }

    @Override
    public int outputSize(int inputSize) {
        return base.outputSize(inputSize);
    }

    /** @return a plain {@code Dense} with the adapted weight folded in */
    public Dense merge() {
        Tensor w = weight().value();
        return new Dense(w.shape().dim(0), w.shape().dim(1), base.activation(), w, bias);
    }

    @Override
    public String toString() {
        return "DoRA(" + base + ", rank " + a.shape().dim(1) + ")";
    }
}
