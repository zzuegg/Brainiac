package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * A reversible residual block (RevNet, Gomez et al. 2017,
 * <a href="https://arxiv.org/abs/1707.04585">arXiv:1707.04585</a>): the features split into
 * halves {@code (x1, x2)} and pass through additive coupling, {@code y1 = x1 + F(x2)},
 * {@code y2 = x2 + G(y1)}. The input is recomputable exactly from the output,
 * {@code x2 = y2 - G(y1)}, {@code x1 = y1 - F(x2)}, so a {@link #stack} of blocks keeps no
 * activations at all for the backward pass: it runs forward without a graph and, going
 * backwards, reconstructs each block's input and back-propagates through that block alone.
 * Activation memory is constant in depth; the price is one extra forward pass of {@code F} and
 * {@code G} (the same as checkpointing every block) and floating-point drift in the
 * reconstruction, measured by {@link #reconstructionError}. The same coupling gives reversible
 * transformers (Reformer).
 *
 * <p>Inputs are {@code [batch, features]} with an even feature count. A single block is an
 * ordinary {@link Layer} (with a normal graph); stacking is where the
 * memory goes away.
 */
public final class ReversibleBlock implements Layer {

    private final Layer f;
    private final Layer g;

    /**
     * @param f the first coupling function, half width to half width
     * @param g the second coupling function, half width to half width
     */
    public ReversibleBlock(Layer f, Layer g) {
        this.f = f;
        this.g = g;
    }

    /**
     * A block whose coupling functions are one tanh dense layer each.
     *
     * @param half the half width
     * @param random the initializer's random source
     * @return the block
     */
    public static ReversibleBlock dense(int half, SplittableRandom random) {
        return new ReversibleBlock(new Dense(half, half, Activation.TANH, Init.XAVIER, random),
                new Dense(half, half, Activation.TANH, Init.XAVIER, random));
    }

    /**
     * A spec for a stack of dense-coupled blocks on an even feature width.
     *
     * @param depth the number of blocks
     * @return the spec
     */
    public static LayerSpec spec(int depth) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                List<ReversibleBlock> blocks = new ArrayList<>();
                for (int k = 0; k < depth; k++) {
                    blocks.add(dense(input.dim(-1) / 2, random));
                }
                return stack(blocks, UnaryOperator.identity());
            }
        };
    }

    @Override
    public Var forward(Var input) {
        int half = input.shape().dim(1) / 2;
        Var x1 = input.slice(1, 0, half);
        Var x2 = input.slice(1, half, 2 * half);
        Var y1 = x1.add(f.forward(x2));
        Var y2 = x2.add(g.forward(y1));
        return Var.concat(1, y1, y2);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(f.parameters());
        p.addAll(g.parameters());
        return p;
    }

    private Tensor forwardValue(Tensor x, UnaryOperator<Tensor> precision) {
        int half = x.shape().dim(1) / 2;
        Tensor x1 = x.slice(1, 0, half);
        Tensor x2 = x.slice(1, half, 2 * half);
        Tensor y1 = precision.apply(x1.add(f.forward(Var.of(x2)).value()));
        Tensor y2 = precision.apply(x2.add(g.forward(Var.of(y1)).value()));
        return Tensor.concat(1, y1, y2);
    }

    private Tensor inverseValue(Tensor y, UnaryOperator<Tensor> precision) {
        int half = y.shape().dim(1) / 2;
        Tensor y1 = y.slice(1, 0, half);
        Tensor y2 = y.slice(1, half, 2 * half);
        Tensor x2 = precision.apply(y2.sub(g.forward(Var.of(y1)).value()));
        Tensor x1 = precision.apply(y1.sub(f.forward(Var.of(x2)).value()));
        return Tensor.concat(1, x1, x2);
    }

    /**
     * A stack of blocks that stores no intermediate activations: its backward pass reconstructs
     * them block by block from the output.
     *
     * @param blocks the blocks, in forward order
     * @param precision applied to every block output and reconstruction, for example a bf16
     *     rounding to study drift; identity for fp32
     * @return the stack as one layer
     */
    public static Layer stack(List<ReversibleBlock> blocks, UnaryOperator<Tensor> precision) {
        List<ReversibleBlock> list = List.copyOf(blocks);
        List<Var> params = new ArrayList<>();
        for (ReversibleBlock b : list) {
            params.addAll(b.parameters());
        }
        return new Layer() {
            @Override
            public Var forward(Var input) {
                Var[] inputs = new Var[1 + params.size()];
                inputs[0] = input;
                for (int i = 0; i < params.size(); i++) {
                    inputs[1 + i] = params.get(i);
                }
                return Var.custom(in -> Var.noGrad(() -> {
                    Tensor y = in[0];
                    for (ReversibleBlock b : list) {
                        y = b.forwardValue(y, precision);
                    }
                    return y;
                }), (in, out, grad) -> backward(list, params, precision, out, grad), inputs);
            }

            @Override
            public List<Var> parameters() {
                return List.copyOf(params);
            }

            @Override
            public String toString() {
                return "ReversibleStack(" + list.size() + " blocks)";
            }
        };
    }

    private static Tensor[] backward(List<ReversibleBlock> blocks, List<Var> params, UnaryOperator<Tensor> precision,
            Tensor out, Tensor grad) {
        Tensor[] result = new Tensor[1 + params.size()];
        int paramEnd = params.size();
        Tensor y = out;
        Tensor gy = grad;
        for (int k = blocks.size() - 1; k >= 0; k--) {
            ReversibleBlock b = blocks.get(k);
            int half = y.shape().dim(1) / 2;
            Tensor y1 = y.slice(1, 0, half);
            Tensor y2 = y.slice(1, half, 2 * half);
            Tensor gy1 = gy.slice(1, 0, half);
            Tensor gy2 = gy.slice(1, half, 2 * half);
            List<Var> gp = b.g.parameters();
            List<Var> fp = b.f.parameters();
            Var z1 = Var.param(y1);
            Var gOut = b.g.forward(z1);
            List<Var> wrtG = new ArrayList<>();
            wrtG.add(z1);
            wrtG.addAll(gp);
            List<Var> dG = Var.gradients(gOut, Var.of(gy2), wrtG, false);
            Tensor gz1 = gy1.add(dG.get(0).value());
            Tensor x2 = precision.apply(y2.sub(gOut.value()));
            Var x2v = Var.param(x2);
            Var fOut = b.f.forward(x2v);
            List<Var> wrtF = new ArrayList<>();
            wrtF.add(x2v);
            wrtF.addAll(fp);
            List<Var> dF = Var.gradients(fOut, Var.of(gz1), wrtF, false);
            Tensor x1 = precision.apply(y1.sub(fOut.value()));
            Tensor gx2 = gy2.add(dF.get(0).value());
            // parameters are laid out block by block, F's then G's
            int start = paramEnd - fp.size() - gp.size();
            for (int i = 0; i < fp.size(); i++) {
                result[1 + start + i] = dF.get(1 + i).value();
            }
            for (int i = 0; i < gp.size(); i++) {
                result[1 + start + fp.size() + i] = dG.get(1 + i).value();
            }
            paramEnd = start;
            y = Tensor.concat(1, x1, x2);
            gy = Tensor.concat(1, gz1, gx2);
        }
        result[0] = gy;
        return result;
    }

    /**
     * The largest absolute difference between an input and its reconstruction from the
     * stack's output: the numerical drift of reversibility.
     *
     * @param blocks the stack
     * @param x an input
     * @param precision the rounding applied to every block output and reconstruction
     * @return {@code max |x - inverse(forward(x))|}
     */
    public static double reconstructionError(List<ReversibleBlock> blocks, Tensor x, UnaryOperator<Tensor> precision) {
        return Var.noGrad(() -> {
            Tensor y = precision.apply(x);
            Tensor start = y;
            for (ReversibleBlock b : blocks) {
                y = b.forwardValue(y, precision);
            }
            for (int k = blocks.size() - 1; k >= 0; k--) {
                y = blocks.get(k).inverseValue(y, precision);
            }
            return (double) y.sub(start).abs().max().item();
        });
    }

    @Override
    public String toString() {
        return "ReversibleBlock(" + f + ", " + g + ")";
    }
}
