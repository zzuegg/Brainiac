package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A {@link ConditionalDenoiser} built on {@link UNet2d}, the classic diffusion denoiser: the
 * flattened sample {@code [batch, side * side]} is reshaped into a one-channel image, and the
 * sinusoidal time features plus a one-hot class (with a "no label" slot for classifier-free
 * guidance) condition every U-Net stage through per-channel biases. A drop-in for
 * {@link MlpDenoiser} and {@link DiffusionTransformer} in core {@code Diffusion} and the
 * flow-matching harnesses.
 *
 * <p>Paper: Ho, Jain and Abbeel, "Denoising Diffusion Probabilistic Models" (2020),
 * <a href="https://arxiv.org/abs/2006.11239">arXiv:2006.11239</a>, which introduced the U-Net
 * denoiser; the U-Net is Ronneberger et al., <a href="https://arxiv.org/abs/1505.04597">arXiv:1505.04597</a>.
 */
public final class UNetDenoiser implements ConditionalDenoiser {
    private static final int TIME_FEATURES = 16;
    private final int side;
    private final int classes;
    private final UNet2d net;
    private float timeScale = 1;

    private UNetDenoiser(int side, int classes, int base, int levels, long seed) {
        this.side = side;
        this.classes = classes;
        this.net = new UNet2d(1, 1, base, levels, true, TIME_FEATURES + classes + 1, new SplittableRandom(seed));
    }

    /**
     * Creates the denoiser.
     *
     * @param side the image side; divisible by {@code 2^levels}
     * @param classes the number of classes, 0 for unconditional only
     * @param base the U-Net's first-stage channels
     * @param levels the U-Net's downsampling stages
     * @param seed the initialization seed
     * @return the denoiser
     */
    public static UNetDenoiser of(int side, int classes, int base, int levels, long seed) {
        return new UNetDenoiser(side, classes, base, levels, seed);
    }

    /**
     * Sets the factor applied to incoming times so they land in about [0, 1]: {@code 1 / steps}
     * for core {@code Diffusion}, 1 (the default) for flow matching.
     *
     * @param scale the factor
     * @return this denoiser
     */
    public UNetDenoiser timeScale(float scale) {
        this.timeScale = scale;
        return this;
    }

    @Override
    public Var apply(Var noisy, Tensor time, int[] labels) {
        float[] t = time.toArray();
        for (int i = 0; i < t.length; i++) {
            t[i] *= timeScale;
        }
        Tensor cond = Tensor.concat(1, GenBlocks.timeEmbedding(t, TIME_FEATURES), GenBlocks.oneHot(labels, classes))
                .to(noisy.value().device());
        int b = noisy.shape().dim(0);
        Var out = net.forward(noisy.reshape(b, 1, side, side), Var.of(cond));
        return out.reshape(b, side * side);
    }

    @Override
    public List<Var> parameters() {
        return net.parameters();
    }

    @Override
    public int dimension() {
        return side * side;
    }

    @Override
    public int classes() {
        return classes;
    }
}
