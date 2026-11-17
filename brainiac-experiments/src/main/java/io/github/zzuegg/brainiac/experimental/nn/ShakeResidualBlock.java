package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.BatchNorm;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A residual block over {@code [batch, channels, h, w]} regularised by stochastic branch
 * mixing, with a different random coefficient in the forward and the backward pass:
 *
 * <ul>
 *   <li><b>Shake-Shake</b>: two branches {@code F1, F2} (each ReLU-conv-ReLU-conv-BatchNorm) are mixed
 *       as {@code x + a F1 + (1 - a) F2} with {@code a ~ U(0, 1)} per example in the forward
 *       pass and an independent {@code b ~ U(0, 1)} in the backward pass; inference uses 0.5.</li>
 *   <li><b>ShakeDrop</b>: one branch scaled by {@code g + a - g a} forward and {@code g + b - g b}
 *       backward, with {@code g ~ Bernoulli(keep)}, {@code a ~ U(-1, 1)} and {@code b ~ U(0, 1)}:
 *       kept branches are untouched, the others are shaken (even negated); inference uses
 *       {@code keep}.</li>
 * </ul>
 *
 * <p>Each branch ends in a core {@code BatchNorm}, as in both papers: without it the mismatch
 * between the forward and backward coefficients feeds back and the weights diverge.
 *
 * <p>The forward/backward split is a straight-through trick:
 * {@code y = c_b F + detach((c_f - c_b) F)} has the value {@code c_f F} and the gradient of
 * {@code c_b F}.
 *
 * <p>Papers: Gastaldi, "Shake-Shake regularization" (2017),
 * <a href="https://arxiv.org/abs/1705.07485">arXiv:1705.07485</a>; Yamada et al., "ShakeDrop
 * Regularization for Deep Residual Learning" (2018),
 * <a href="https://arxiv.org/abs/1802.02375">arXiv:1802.02375</a>.
 */
public final class ShakeResidualBlock implements Layer {

    /** The mixing rule. */
    public enum Mode {
        /** two branches, random convex mix */
        SHAKE_SHAKE,
        /** one branch, random scale with Bernoulli gate */
        SHAKE_DROP
    }

    private final int channels;
    private final Mode mode;
    private final float keep;
    private final List<Var[]> branches = new ArrayList<>();
    private final List<BatchNorm> norms = new ArrayList<>();
    private final RandomGenerator random;

    /**
     * Creates the block.
     *
     * @param channels the channels
     * @param mode the mixing rule
     * @param keep for ShakeDrop, the probability that the branch is left unshaken
     * @param random the source of weights and coefficients
     */
    public ShakeResidualBlock(int channels, Mode mode, float keep, RandomGenerator random) {
        this.channels = channels;
        this.mode = mode;
        this.keep = keep;
        this.random = random;
        int n = mode == Mode.SHAKE_SHAKE ? 2 : 1;
        for (int i = 0; i < n; i++) {
            Var w2 = Var.param(Tensor.normal(Shape.of(channels, channels, 3, 3), 0,
                    (float) (0.5 * Math.sqrt(2.0 / (9 * channels))), random));
            branches.add(new Var[] {ClassicOps.convWeight(random, channels, channels, 3), ClassicOps.zeros(channels),
                w2, ClassicOps.zeros(channels)});
            norms.add(new BatchNorm(channels));
        }
    }

    /**
     * Describes the block for model builders.
     *
     * @param mode the mixing rule
     * @param keep the ShakeDrop keep probability (ignored by Shake-Shake)
     * @return the spec
     */
    public static LayerSpec spec(Mode mode, float keep) {
        return TechniqueOps.sameShape((s, r) -> new ShakeResidualBlock(s.dim(0), mode, keep, r));
    }

    private Var branch(Var x, int i, boolean training) {
        Var[] w = branches.get(i);
        Var h = ClassicOps.conv(x.relu(), w[0], w[1], 1);
        return norms.get(i).forward(ClassicOps.conv(h.relu(), w[2], w[3], 1), training);
    }

    private Tensor perExample(int b, float low, float high, Tensor like) {
        return ClassicOps.uniform(Shape.of(b, 1, 1, 1), low, high, random, like.device());
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        ClassicOps.requireImages(input, channels, this);
        int b = input.shape().dim(0);
        if (mode == Mode.SHAKE_SHAKE) {
            Var f1 = branch(input, 0, training);
            Var f2 = branch(input, 1, training);
            if (!training) {
                return input.add(f1.add(f2).mul(0.5f));
            }
            Tensor a = perExample(b, 0, 1, input.value());
            Tensor beta = perExample(b, 0, 1, input.value());
            Var diff = f1.sub(f2);
            // forward: f2 + a (f1 - f2); backward: f2 + beta (f1 - f2)
            Var mixed = f2.add(diff.mul(beta)).add(diff.mul(a.sub(beta)).detach());
            return input.add(mixed);
        }
        Var f = branch(input, 0, training);
        if (!training) {
            return input.add(f.mul(keep));
        }
        Tensor gate = ClassicOps.bernoulli(Shape.of(b, 1, 1, 1), keep, random, input.value().device());
        Tensor a = perExample(b, -1, 1, input.value());
        Tensor beta = perExample(b, 0, 1, input.value());
        Tensor forwardScale = gate.add(a).sub(gate.mul(a));
        Tensor backwardScale = gate.add(beta).sub(gate.mul(beta));
        return input.add(f.mul(backwardScale).add(f.mul(forwardScale.sub(backwardScale)).detach()));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < branches.size(); i++) {
            p.addAll(List.of(branches.get(i)));
            p.addAll(norms.get(i).parameters());
        }
        return p;
    }

    @Override
    public List<Var> buffers() {
        List<Var> b = new ArrayList<>();
        for (BatchNorm n : norms) {
            b.addAll(n.buffers());
        }
        return b;
    }

    @Override
    public String toString() {
        return "ShakeResidualBlock(" + channels + ", " + mode + ")";
    }
}
