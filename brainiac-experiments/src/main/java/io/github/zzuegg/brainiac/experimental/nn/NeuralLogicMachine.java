package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A Neural Logic Machine over unary and binary predicates. Each layer holds fuzzy
 * predicate tensors of arity 1 ({@code [n, c1]}) and 2 ({@code [n, n, c2]}) and computes
 * new ones with position-wise MLPs from:
 * <ul>
 *   <li>the same arity's predicates and, for arity 2, their transposes (argument
 *       permutation);</li>
 *   <li>the arity below, expanded ({@code U(x)} and {@code U(y)} broadcast to pairs);</li>
 *   <li>the arity above, reduced by the quantifiers {@code exists} (max) and
 *       {@code forall} (min) over either argument.</li>
 * </ul>
 * Because every operation is shared across objects, a machine trained on small worlds runs
 * on larger ones, and stacking layers composes rules (depth bounds the chain length it can
 * express). Input: background binary relations; output: logits of a binary target
 * {@code [n, n]}.
 *
 * <p>Paper: Dong, Mao, Lin, Wang, Li, Zhou, "Neural Logic Machines" (ICLR 2019),
 * <a href="https://arxiv.org/abs/1904.11694">arXiv:1904.11694</a>.
 */
public final class NeuralLogicMachine {

    private final int depth;
    private final List<Var[]> unary = new ArrayList<>();    // {W, b}
    private final List<Var[]> binary = new ArrayList<>();
    private final Var out;
    private final Var outBias;

    /**
     * Creates the machine.
     *
     * @param relations input binary relations
     * @param hidden predicates per arity per layer
     * @param depth layers
     * @param seed the seed
     */
    public NeuralLogicMachine(int relations, int hidden, int depth, long seed) {
        this.depth = depth;
        SplittableRandom r = new SplittableRandom(seed);
        int c1 = 1;
        int c2 = relations + 1;   // plus the identity relation x == y
        for (int l = 0; l < depth; l++) {
            int in1 = c1 + 4 * c2;
            int in2 = 2 * c2 + 2 * c1;
            unary.add(new Var[] {MixerOps.weight(r, in1, hidden), MixerOps.filled(0f, hidden)});
            binary.add(new Var[] {MixerOps.weight(r, in2, hidden), MixerOps.filled(0f, hidden)});
            c1 = hidden;
            c2 = hidden;
        }
        out = MixerOps.weight(r, c2, 1);
        outBias = MixerOps.filled(0f, 1);
    }

    /**
     * Derives the target.
     *
     * @param relations the background relations, each {@code [n, n]}
     * @return target logits {@code [n, n]}
     */
    public Var forward(List<Tensor> relations) {
        int n = relations.get(0).shape().dim(0);
        float[] eye = new float[n * n];
        for (int i = 0; i < n; i++) {
            eye[i * n + i] = 1;
        }
        List<Var> rels = new ArrayList<>();
        for (Tensor t : relations) {
            rels.add(Var.of(t.reshape(n, n, 1)));
        }
        rels.add(Var.of(Tensor.of(Shape.of(n, n, 1), eye)));
        Var b = Var.concat(2, rels.toArray(Var[]::new));
        Var u = Var.of(Tensor.full(Shape.of(n, 1), 1f));
        for (int l = 0; l < depth; l++) {
            int c1 = u.shape().dim(1);
            Var bt = b.permute(1, 0, 2);
            Var reduced = Var.concat(1, u, b.max(1, false), b.neg().max(1, false).neg(), bt.max(1, false),
                    bt.neg().max(1, false).neg());
            Var ux = u.reshape(n, 1, c1).expand(Shape.of(n, n, c1));
            Var uy = u.reshape(1, n, c1).expand(Shape.of(n, n, c1));
            Var pairs = Var.concat(2, b, bt, ux, uy);
            Var[] wu = unary.get(l);
            Var[] wb = binary.get(l);
            u = reduced.matmul(wu[0]).add(wu[1]).sigmoid();
            b = pairs.matmul(wb[0]).add(wb[1]).sigmoid();
        }
        return b.matmul(out).add(outBias).reshape(n, n);
    }

    /**
     * Class-balanced binary cross-entropy against a target relation: positive pairs are
     * weighted by {@code negatives / positives}, since target relations are sparse.
     *
     * @param relations the background relations
     * @param target the target {@code [n, n]}
     * @return the mean loss
     */
    public Var loss(List<Tensor> relations, Tensor target) {
        Var z = forward(relations);
        float pos = target.sum().item();
        float total = target.shape().size();
        float weight = pos == 0 ? 1 : Math.min(50, (total - pos) / pos);
        // -[w y log s(z) + (1 - y) log(1 - s(z))] = w y softplus(-z) + (1 - y) softplus(z)
        Var positive = z.neg().softplus().mul(target).mul(weight);
        Var negative = z.softplus().mul(target.neg().add(1f));
        return positive.add(negative).mean();
    }

    /** @return all weights */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        unary.forEach(w -> p.addAll(List.of(w)));
        binary.forEach(w -> p.addAll(List.of(w)));
        p.add(out);
        p.add(outBias);
        return p;
    }
}
