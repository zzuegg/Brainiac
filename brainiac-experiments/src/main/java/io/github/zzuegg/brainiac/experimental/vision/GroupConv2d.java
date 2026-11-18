package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A p4 group convolution (G-CNN, Cohen and Welling, 2016): convolution that is equivariant
 * to translations and to rotations by multiples of 90 degrees. A feature map on p4 has four
 * channels per field, one per rotation {@code r}; the channel of field {@code f} and rotation
 * {@code r} is {@code 4 f + r}.
 *
 * <ul>
 *   <li>{@link Kind#LIFT} takes an ordinary image and convolves it with the four rotated
 *       copies {@code R^r w} of every filter.</li>
 *   <li>{@link Kind#GROUP} takes a p4 map: the filter of output rotation {@code r} is the
 *       base filter rotated by {@code r} in space and cyclically shifted by {@code r} along
 *       its own rotation axis.</li>
 * </ul>
 *
 * <p>Rotating the input by 90 degrees then rotates every output map and shifts the rotation
 * channels by one; {@link #poolRotations} (max over the four rotations) followed by global
 * pooling is therefore exactly invariant. A G-CNN layer has the parameters of one ordinary
 * filter bank but four times the output channels.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1602.07576">arXiv:1602.07576</a>.
 */
public final class GroupConv2d implements Layer {

    /** The input kind. */
    public enum Kind {
        /** an ordinary image to a p4 map */
        LIFT,
        /** a p4 map to a p4 map */
        GROUP
    }

    private final Kind kind;
    private final int inputs;
    private final int outputs;
    private final int kernel;
    private final Var weight;
    private final Var bias;

    /**
     * Creates the layer.
     *
     * @param kind lifting or group convolution
     * @param inputs input channels (LIFT) or input fields of four channels each (GROUP)
     * @param outputs output fields
     * @param kernel the odd kernel side
     * @param random the source of initial weights
     */
    public GroupConv2d(Kind kind, int inputs, int outputs, int kernel, RandomGenerator random) {
        this.kind = kind;
        this.inputs = inputs;
        this.outputs = outputs;
        this.kernel = kernel;
        int fanIn = inputs * kernel * kernel * (kind == Kind.GROUP ? 4 : 1);
        Var w = VisionOps.convWeight(random, outputs, fanIn / (kernel * kernel), kernel);
        this.weight = kind == Kind.GROUP
                ? Var.param(w.value().reshape(Shape.of(outputs, inputs, 4, kernel, kernel))) : w;
        this.bias = VisionOps.zeros(outputs);
    }

    /**
     * Rotates the last two axes by 90 degrees: {@code out[y][x] = in[x][k - 1 - y]}.
     *
     * @param w a value whose last two axes are square
     * @return the rotated value
     */
    public static Var rotate(Var w) {
        int rank = w.shape().rank();
        int[] axes = new int[rank];
        for (int i = 0; i < rank; i++) {
            axes[i] = i;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return w.permute(axes).flip(rank - 2);
    }

    /**
     * Rotates images by {@code times} quarter turns, the way {@link #rotate} rotates filters.
     *
     * @param images {@code [n, c, s, s]}
     * @param times quarter turns
     * @return the rotated images
     */
    public static Tensor rotateImages(Tensor images, int times) {
        Var v = Var.of(images);
        for (int t = 0; t < Math.floorMod(times, 4); t++) {
            v = rotate(v);
        }
        return v.value();
    }

    /**
     * Max-pools a p4 map over its four rotations.
     *
     * @param x {@code [batch, 4 * fields, h, w]}
     * @return {@code [batch, fields, h, w]}
     */
    public static Var poolRotations(Var x) {
        Shape s = x.shape();
        return x.reshape(s.dim(0), s.dim(1) / 4, 4, s.dim(2), s.dim(3)).max(2, false);
    }

    /** @return the full filter bank, {@code [4 * outputs, channels, k, k]} */
    public Var filters() {
        Var[] perRotation = new Var[4];
        for (int r = 0; r < 4; r++) {
            Var w = weight;
            if (kind == Kind.GROUP && r > 0) {
                w = Var.concat(2, weight.slice(2, 4 - r, 4), weight.slice(2, 0, 4 - r));
            }
            for (int t = 0; t < r; t++) {
                w = rotate(w);
            }
            int in = kind == Kind.GROUP ? 4 * inputs : inputs;
            perRotation[r] = w.reshape(outputs, 1, in, kernel, kernel);
        }
        int in = kind == Kind.GROUP ? 4 * inputs : inputs;
        return Var.concat(1, perRotation).reshape(4 * outputs, in, kernel, kernel);
    }

    @Override
    public Var forward(Var input) {
        return VisionOps.conv(input, filters(), bias.repeatInterleave(4, 0));
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(4 * outputs, input.dim(1), input.dim(2));
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public String toString() {
        return "GroupConv2d(" + kind + ", " + inputs + " -> " + outputs + ", " + kernel + "x" + kernel + ")";
    }
}
