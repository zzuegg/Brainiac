package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * SchNet (Schutt et al., 2017): a rotation- and translation-invariant energy model for atoms.
 * Every atom starts from a learned embedding; interaction blocks update it with a
 * {@link ContinuousFilterConv} over pair distances,
 *
 * <pre>
 * v = dense(ssp(dense(cfconv(dense(x), d))));   x = x + v
 * E = sum_i out(x_i)
 * </pre>
 *
 * <p>and forces are the negative gradient of the energy with respect to positions, taken by
 * autograd ({@link #forces}), so the force field is conservative. {@link #loss} trains on
 * energies and, with a force weight, on forces as well (force matching), which supplies
 * {@code 3n} labels per configuration instead of one.
 *
 * <p>As a {@link Layer}, {@code forward} maps positions {@code [b, n * 3]} to energies
 * {@code [b, 1]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1706.08566">arXiv:1706.08566</a>.
 */
public final class SchNet implements Layer {

    private final int features;
    private final Var embedding;
    private final List<ContinuousFilterConv> convs = new ArrayList<>();
    private final List<Var[]> dense = new ArrayList<>();
    private final Var out1;
    private final Var out1b;
    private final Var out2;
    private final Var out2b;

    private SchNet(int features, int interactions, int basis, double cutoff, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.features = features;
        this.embedding = Var.param(Tensor.normal(Shape.of(1, 1, features), 0, 1f, r));
        for (int i = 0; i < interactions; i++) {
            convs.add(ContinuousFilterConv.of(features, basis, cutoff, r.nextLong()));
            dense.add(new Var[] {weight(features, features, r), weight(features, features, r),
                Var.param(Tensor.zeros(Shape.of(features))), weight(features, features, r),
                Var.param(Tensor.zeros(Shape.of(features)))});
        }
        this.out1 = weight(features, features / 2, r);
        this.out1b = Var.param(Tensor.zeros(Shape.of(features / 2)));
        this.out2 = weight(features / 2, 1, r);
        this.out2b = Var.param(Tensor.zeros(Shape.of(1)));
    }

    private static Var weight(int in, int out, SplittableRandom r) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(1.0 / in), r));
    }

    /**
     * Creates the model.
     *
     * @param features atom feature width
     * @param interactions interaction blocks
     * @param basis radial basis functions
     * @param cutoff the cutoff distance
     * @param seed the seed
     * @return the model
     */
    public static SchNet of(int features, int interactions, int basis, double cutoff, long seed) {
        return new SchNet(features, interactions, basis, cutoff, seed);
    }

    /**
     * Pairwise distances, smooth everywhere (a small epsilon under the square root).
     *
     * @param positions {@code [b, n, 3]}
     * @return {@code [b, n, n]}
     */
    public static Var distances(Var positions) {
        int b = positions.shape().dim(0);
        int n = positions.shape().dim(1);
        Shape s = Shape.of(b, n, n, 3);
        Var diff = positions.reshape(b, n, 1, 3).expand(s).sub(positions.reshape(b, 1, n, 3).expand(s));
        return diff.square().sum(3, false).add(1e-6f).sqrt();
    }

    /**
     * Per-atom energies.
     *
     * @param positions {@code [b, n, 3]}
     * @return {@code [b, n, 1]}
     */
    public Var atomEnergies(Var positions) {
        int b = positions.shape().dim(0);
        int n = positions.shape().dim(1);
        Var d = distances(positions);
        Var x = embedding.expand(Shape.of(b, n, features));
        for (int i = 0; i < convs.size(); i++) {
            Var[] w = dense.get(i);
            Var v = convs.get(i).forward(x.matmul(w[0]), d);
            v = ContinuousFilterConv.shiftedSoftplus(v.matmul(w[1]).add(w[2])).matmul(w[3]).add(w[4]);
            x = x.add(v);
        }
        return ContinuousFilterConv.shiftedSoftplus(x.matmul(out1).add(out1b)).matmul(out2).add(out2b);
    }

    /**
     * The total energy.
     *
     * @param positions {@code [b, n, 3]}
     * @return {@code [b, 1]}
     */
    public Var energy(Var positions) {
        return atomEnergies(positions).sum(1, false);
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        return energy(input.reshape(b, input.shape().dim(1) / 3, 3));
    }

    /**
     * Forces {@code -dE/dx}.
     *
     * @param positions {@code [b, n, 3]} as a value with gradient ({@code Var.param})
     * @param createGraph whether the forces stay differentiable (for force-matching training)
     * @return {@code [b, n, 3]}
     */
    public Var forces(Var positions, boolean createGraph) {
        return Var.gradients(energy(positions).sum(), List.of(positions), createGraph).get(0).neg();
    }

    /**
     * The training loss on energies and optionally forces.
     *
     * @param positions {@code [b, n, 3]}
     * @param energies targets {@code [b, 1]}
     * @param forces targets {@code [b, n, 3]}, or {@code null}
     * @param forceWeight the weight of the force term
     * @return the loss
     */
    public Var loss(Tensor positions, Tensor energies, Tensor forces, float forceWeight) {
        Var p = Var.param(positions);
        Var e = energy(p);
        Var loss = e.sub(energies).square().mean();
        if (forces != null && forceWeight > 0) {
            Var f = Var.gradients(e.sum(), List.of(p), true).get(0).neg();
            loss = loss.add(f.sub(forces).square().mean().mul(forceWeight));
        }
        return loss;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding, out1, out1b, out2, out2b));
        for (int i = 0; i < convs.size(); i++) {
            p.addAll(convs.get(i).parameters());
            p.addAll(List.of(dense.get(i)));
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return 1;
    }

    @Override
    public String toString() {
        return "SchNet(F=" + features + ", interactions=" + convs.size() + ")";
    }
}
