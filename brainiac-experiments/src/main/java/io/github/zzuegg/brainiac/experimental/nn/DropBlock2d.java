package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * DropBlock over feature maps {@code [batch, channels, height, width]}: instead of dropping
 * independent pixels (which neighbouring, correlated activations simply fill back in), it
 * drops contiguous {@code block x block} squares. Seeds are drawn with rate
 * {@code gamma = rate / block^2 * (h w) / ((h - block + 1)(w - block + 1))} so that about a
 * {@code rate} share of the units is removed, each seed grows into a block, and the survivors
 * are rescaled by {@code count / kept}. With {@code block = 1} it is ordinary dropout; with a
 * block as large as the map and shared channels it approaches spatial dropout.
 *
 * <p>Paper: Ghiasi, Lin and Le, "DropBlock: A regularization method for convolutional
 * networks" (NeurIPS 2018), <a href="https://arxiv.org/abs/1810.12890">arXiv:1810.12890</a>.
 */
public final class DropBlock2d implements Layer {

    private final float rate;
    private final int block;
    private final boolean sharedChannels;
    private final RandomGenerator random;

    /**
     * Creates the layer.
     *
     * @param rate the share of units to drop, in {@code [0, 1)}
     * @param block the block side
     * @param sharedChannels whether all channels of an example share one mask
     * @param random the mask source
     */
    public DropBlock2d(float rate, int block, boolean sharedChannels, RandomGenerator random) {
        if (!(rate >= 0 && rate < 1) || block < 1) {
            throw new IllegalArgumentException("need rate in [0, 1) and block >= 1, got " + rate + ", " + block);
        }
        this.rate = rate;
        this.block = block;
        this.sharedChannels = sharedChannels;
        this.random = random;
    }

    /**
     * Describes the layer for model builders.
     *
     * @param rate the drop rate
     * @param block the block side
     * @return the spec
     */
    public static LayerSpec spec(float rate, int block) {
        return TechniqueOps.sameShape((s, r) -> new DropBlock2d(rate, block, false, r));
    }

    @Override
    public Var forward(Var input) {
        return input;
    }

    @Override
    public Var forward(Var input, boolean training) {
        ClassicOps.requireImages(input, 0, this);
        if (!training || rate == 0) {
            return input;
        }
        return input.mul(mask(input.shape()).to(input.value().device()));
    }

    /**
     * Draws one rescaled keep mask.
     *
     * @param s the feature map shape {@code [batch, channels, h, w]}
     * @return the mask, broadcastable to {@code s}
     */
    Tensor mask(Shape s) {
        int b = s.dim(0);
        int c = sharedChannels ? 1 : s.dim(1);
        int h = s.dim(2);
        int w = s.dim(3);
        int bs = Math.min(block, Math.min(h, w));
        double gamma = rate / (bs * bs) * (h * w) / ((double) (h - bs + 1) * (w - bs + 1));
        float[] m = new float[b * c * h * w];
        java.util.Arrays.fill(m, 1f);
        int lo = bs / 2;
        for (int i = 0; i < b * c; i++) {
            int o = i * h * w;
            for (int y = lo; y < h - (bs - 1 - lo); y++) {
                for (int x = lo; x < w - (bs - 1 - lo); x++) {
                    if (random.nextDouble() < gamma) {
                        for (int dy = 0; dy < bs; dy++) {
                            for (int dx = 0; dx < bs; dx++) {
                                m[o + (y - lo + dy) * w + (x - lo + dx)] = 0f;
                            }
                        }
                    }
                }
            }
        }
        double kept = 0;
        for (float v : m) {
            kept += v;
        }
        float scale = kept == 0 ? 0f : (float) (m.length / kept);
        for (int i = 0; i < m.length; i++) {
            m[i] *= scale;
        }
        return Tensor.of(Shape.of(b, c, h, w), m);
    }

    @Override
    public String toString() {
        return "DropBlock2d(rate " + rate + ", block " + block + ")";
    }
}
