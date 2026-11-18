package io.github.zzuegg.brainiac.experimental.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * TinyTL's lite residual branches: with the backbone weights frozen and only biases trained (bias
 * updates need no stored activations), small side branches restore capacity at little memory.
 * Each branch takes a convolution's input, average-pools it 2x2 (a quarter of the activations to
 * store), applies a 3x3 group convolution and ReLU, then a 1x1 convolution (zero-initialised, so
 * training starts from the backbone) and upsamples to the convolution's output, which it is
 * added to. Attached through {@link EdgeNet#setBranch}.
 *
 * <p>Paper: Cai et al., "TinyTL: Reduce Memory, Not Parameters for Efficient On-Device
 * Learning", NeurIPS 2020, <a href="https://arxiv.org/abs/2007.11622">arXiv:2007.11622</a>.
 */
public final class LiteResidual {

    private final EdgeNet net;
    private final int[] units;
    private final List<Var> parameters = new ArrayList<>();
    private final long storedElements;

    /**
     * Attaches branches.
     *
     * @param net the network
     * @param units convolution units to add branches to
     * @param groupWidth channels per group of the 3x3 convolution
     * @param seed the seed
     */
    public LiteResidual(EdgeNet net, int[] units, int groupWidth, long seed) {
        this.net = net;
        this.units = units.clone();
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> shapes = net.shapes();
        long stored = 0;
        for (int i : units) {
            EdgeNet.Conv c = (EdgeNet.Conv) net.units().get(i);
            int cin = c.inChannels();
            int groups = Math.max(1, cin / Math.max(1, groupWidth));
            while (cin % groups != 0) {
                groups--;
            }
            int g = groups;
            int[] inShape = i == 0 ? net.inputShape() : shapes.get(i - 1);
            int[] outShape = shapes.get(i);
            Var w1 = Var.param(Tensor.normal(Shape.of(cin, cin / g, 3, 3), 0, (float) Math.sqrt(2.0 / (9 * cin / g)), r));
            Var b1 = Var.param(Tensor.zeros(Shape.of(cin)));
            Var w2 = Var.param(Tensor.zeros(Shape.of(c.outChannels(), cin, 1, 1)));
            Var b2 = Var.param(Tensor.zeros(Shape.of(c.outChannels())));
            parameters.addAll(List.of(w1, b1, w2, b2));
            long pooled = (long) cin * (inShape[1] / 2) * (inShape[2] / 2);
            stored += 2 * pooled;     // the pooled input and the group convolution's output
            int[] size = {outShape[1], outShape[2]};
            net.setBranch(i, x -> {
                Var p = Functional.pool(x, Functional.PoolKind.AVG, new int[] {2, 2}, new int[] {2, 2},
                        new int[] {0, 0}, new int[] {0, 0}, new int[] {1, 1}, false, false, 1);
                Var h = Functional.conv(p, w1, b1, new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1},
                        new int[] {1, 1}, g).relu();
                h = Functional.conv(h, w2, b2, new int[] {1, 1}, new int[] {0, 0}, new int[] {0, 0},
                        new int[] {1, 1}, 1);
                if (h.shape().dim(2) != size[0] || h.shape().dim(3) != size[1]) {
                    h = Functional.resize(h, size, Functional.Interpolation.NEAREST, false);
                }
                return h;
            });
        }
        this.storedElements = stored;
    }

    /** @return the branches' parameters */
    public List<Var> parameters() {
        return parameters;
    }

    /** @return the number of branch parameters */
    public long parameterCount() {
        return parameters.stream().mapToLong(p -> p.value().size()).sum();
    }

    /**
     * Activations the branches keep for their backward pass.
     *
     * @param bytesPerElement 1 for int8
     * @return bytes
     */
    public long storedBytes(int bytesPerElement) {
        return storedElements * bytesPerElement;
    }

    /** Removes the branches from the network. */
    public void detach() {
        for (int i : units) {
            net.setBranch(i, null);
        }
    }
}
