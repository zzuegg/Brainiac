package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.KanLayer;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A deep physics-informed Kolmogorov-Arnold network built from {@link KanLayer}s: the
 * coordinates are lifted to {@code width} channels squashed into the spline grid
 * {@code [-1, 1]}, then pass through KAN blocks. With gating (the residual-gating idea of
 * PirateNets carried over to KANs, as in the 2025 deep PIKAN work), each block is
 *
 * <pre>
 * h_{l+1} = alpha_l tanh(KAN_l(h_l)) + (1 - alpha_l) h_l,   alpha_l = 0 at the start
 * </pre>
 *
 * <p>so a deep stack starts shallow and stays trainable; without gating it is the plain deep
 * PIKAN {@code h_{l+1} = tanh(KAN_l(h_l))} that is reported to train poorly as it gets deep.
 * Spline bases are differentiable piecewise polynomials, so second derivatives for PDE
 * residuals come out of the core autograd.
 *
 * <p>Papers: Liu et al., KAN, <a href="https://arxiv.org/abs/2404.19756">arXiv:2404.19756</a>;
 * deep physics-informed KANs, <a href="https://arxiv.org/abs/2510.23501">arXiv:2510.23501</a>.
 */
public final class PhysicsInformedKan implements Layer {

    private final int outputs;
    private final boolean gated;
    private final Var lift;
    private final Var liftBias;
    private final List<KanLayer> blocks = new ArrayList<>();
    private final List<Var> alphas = new ArrayList<>();
    private final Var output;

    private PhysicsInformedKan(int inputs, int outputs, int width, int depth, int intervals, boolean gated,
            long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        this.outputs = outputs;
        this.gated = gated;
        this.lift = Var.param(Tensor.normal(Shape.of(inputs, width), 0, 1f, random));
        this.liftBias = Var.param(Tensor.normal(Shape.of(1, width), 0, 0.5f, random));
        for (int i = 0; i < depth; i++) {
            blocks.add(KanLayer.of(width, width, intervals, -1, 1, random.nextLong()));
            alphas.add(Var.param(Tensor.zeros(Shape.of(1))));
        }
        this.output = Var.param(Tensor.normal(Shape.of(width, outputs), 0, (float) (1 / Math.sqrt(width)), random));
    }

    /**
     * Creates the network.
     *
     * @param inputs the coordinates
     * @param outputs the field components
     * @param width the channels
     * @param depth the KAN blocks
     * @param intervals spline grid intervals per edge
     * @param gated whether to gate the residual blocks (trainable alpha from 0)
     * @param seed the seed
     * @return the network
     */
    public static PhysicsInformedKan of(int inputs, int outputs, int width, int depth, int intervals, boolean gated,
            long seed) {
        return new PhysicsInformedKan(inputs, outputs, width, depth, intervals, gated, seed);
    }

    @Override
    public Var forward(Var x) {
        Var h = x.matmul(lift).add(liftBias).tanh();
        for (int i = 0; i < blocks.size(); i++) {
            Var k = blocks.get(i).forward(h).tanh();
            if (gated) {
                Var a = alphas.get(i);
                h = k.mul(a).add(h.mul(a.neg().add(1f)));
            } else {
                h = k;
            }
        }
        return h.matmul(output);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(lift, liftBias, output));
        for (int i = 0; i < blocks.size(); i++) {
            p.addAll(blocks.get(i).parameters());
            if (gated) {
                p.add(alphas.get(i));
            }
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public String toString() {
        return "PhysicsInformedKan(blocks=" + blocks.size() + (gated ? ", gated)" : ")");
    }
}
