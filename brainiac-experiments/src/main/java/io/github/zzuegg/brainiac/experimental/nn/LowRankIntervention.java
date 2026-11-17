package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * LoReFT, low-rank linear subspace representation fine-tuning: instead of changing weights,
 * leave the model frozen and edit its hidden representations inside a learned
 * {@code r}-dimensional subspace,
 *
 * <pre>
 * Phi(h) = h + R^T (W h + b - R h)
 * </pre>
 *
 * <p>with {@code R} an {@code r x d} matrix with orthonormal rows. Inside the subspace the
 * representation's coordinates {@code R h} are replaced by learned ones {@code W h + b};
 * outside it nothing changes. This is the distributed interchange intervention of causal
 * abstraction work turned into a fine-tuning method, and it trains 15 to 65 times fewer values
 * than LoRA for similar results in the paper. Here {@code R} and {@code W} are stored as
 * {@code d x r} matrices {@code P} and {@code Q} ({@code Phi(h) = h + (h Q + b - h P) P^T});
 * {@link #orthonormalize()} restores orthonormal columns of {@code P} after each optimizer step
 * (the paper uses an orthogonal parametrization).
 *
 * <p>An intervention can be limited to some positions (LoReFT edits a few prefix and suffix
 * positions of the prompt); input is {@code [..., steps, d]} or {@code [..., d]}. It is a
 * {@link Layer} (with a {@link LayerSpec}), so it goes wherever a layer goes: after a block,
 * as a decoder's adapter slot, or on any step of a core {@link Model} through
 * {@link #hook(int)} ({@link LayerHooks#at}).
 *
 * <p>Wu et al., "ReFT: Representation Finetuning for Language Models" (NeurIPS 2024),
 * <a href="https://arxiv.org/abs/2404.03592">arXiv:2404.03592</a>.
 */
public final class LowRankIntervention implements Layer {

    private final int dim;
    private final int rank;
    private final Var p;
    private final Var q;
    private final Var b;
    private int first;
    private int last;

    /**
     * Creates an intervention that starts as the identity ({@code Q = P}, {@code b = 0}).
     *
     * @param dim the representation size {@code d}
     * @param rank the subspace size {@code r}
     * @param seed seeds the random orthonormal subspace
     */
    public LowRankIntervention(int dim, int rank, long seed) {
        if (rank < 1 || rank > dim) {
            throw new IllegalArgumentException("need 1 <= rank <= dim");
        }
        this.dim = dim;
        this.rank = rank;
        SplittableRandom r = new SplittableRandom(seed);
        float[] m = Tensor.normal(Shape.of(dim, rank), 0, 1, r).toArray();
        gramSchmidt(m, dim, rank);
        this.p = Var.param(Tensor.of(Shape.of(dim, rank), m));
        this.q = Var.param(Tensor.of(Shape.of(dim, rank), m.clone()));
        this.b = Var.param(Tensor.zeros(Shape.of(rank)));
        this.first = -1;
        this.last = -1;
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param rank the subspace size
     * @return the spec
     */
    public static LayerSpec spec(int rank) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new LowRankIntervention(input.dim(input.rank() - 1), rank, random.nextLong());
            }
        };
    }

    /**
     * Intervenes only at the first {@code prefix} and the last {@code suffix} steps of a
     * sequence input (LoReFT's prompt positions); negative values intervene everywhere.
     *
     * @param prefix leading positions
     * @param suffix trailing positions
     * @return this layer
     */
    public LowRankIntervention positions(int prefix, int suffix) {
        this.first = prefix;
        this.last = suffix;
        return this;
    }

    /** @return the subspace size */
    public int rank() {
        return rank;
    }

    @Override
    public Var forward(Var input) {
        Var delta = input.matmul(q).add(b).sub(input.matmul(p)).matmul(p.transpose());
        if (first >= 0 && input.shape().rank() >= 2) {
            int steps = input.shape().dim(-2);
            float[] mask = new float[steps];
            for (int t = 0; t < steps; t++) {
                mask[t] = t < first || t >= steps - last ? 1 : 0;
            }
            delta = delta.mul(Tensor.of(Shape.of(steps, 1), mask));
        }
        return input.add(delta);
    }

    /**
     * A hook that applies this intervention to one step's output of a core model.
     *
     * @param step the step index
     * @return the hook for {@link Model#forward(Var, boolean, Model.Hook)}
     */
    public Model.Hook hook(int step) {
        return LayerHooks.at(step, this::forward);
    }

    /** Re-orthonormalizes the subspace's basis; call after each optimizer step. */
    public void orthonormalize() {
        float[] m = p.value().toArray();
        gramSchmidt(m, dim, rank);
        p.assign(Tensor.of(Shape.of(dim, rank), m, p.value().device()));
    }

    /** Orthonormalizes the columns of a {@code [rows, cols]} row-major matrix in place. */
    private static void gramSchmidt(float[] m, int rows, int cols) {
        for (int c = 0; c < cols; c++) {
            for (int k = 0; k < c; k++) {
                double dot = 0;
                for (int r = 0; r < rows; r++) {
                    dot += m[r * cols + c] * m[r * cols + k];
                }
                for (int r = 0; r < rows; r++) {
                    m[r * cols + c] -= (float) (dot * m[r * cols + k]);
                }
            }
            double norm = 0;
            for (int r = 0; r < rows; r++) {
                norm += m[r * cols + c] * m[r * cols + c];
            }
            norm = Math.sqrt(Math.max(norm, 1e-24));
            for (int r = 0; r < rows; r++) {
                m[r * cols + c] /= (float) norm;
            }
        }
    }

    @Override
    public List<Var> parameters() {
        return List.of(p, q, b);
    }

    @Override
    public int outputSize(int inputSize) {
        return inputSize;
    }

    @Override
    public String toString() {
        return "LowRankIntervention(" + dim + ", rank " + rank + ")";
    }
}
