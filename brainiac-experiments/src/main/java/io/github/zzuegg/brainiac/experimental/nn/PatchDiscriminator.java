package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The PatchGAN discriminator of pix2pix (Isola et al.,
 * <a href="https://arxiv.org/abs/1611.07004">arXiv:1611.07004</a>), as used by CycleGAN and
 * VQ-GAN: a small fully convolutional network that scores every overlapping image patch as
 * real or fake instead of the whole image, so it judges local texture and sharpness while the
 * reconstruction loss handles the global layout. Images {@code [n, channels, h, w]} map to patch
 * logits {@code [n, 1, h / 2, w / 2]}: a stride-2 4x4 convolution, then 3x3 convolutions,
 * leaky ReLU (slope 0.2) between them.
 */
public final class PatchDiscriminator implements Layer {
    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();

    private PatchDiscriminator(int channels, int width, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        add(r, width, channels, 4);
        add(r, 2 * width, width, 3);
        add(r, 1, 2 * width, 3);
    }

    private void add(SplittableRandom r, int out, int in, int kernel) {
        weights.add(TechniqueOps.normal(r, (float) Math.sqrt(2.0 / (in * kernel * kernel)), out, in, kernel, kernel));
        biases.add(TechniqueOps.filled(0f, out));
    }

    /**
     * Creates the discriminator.
     *
     * @param channels the image channels
     * @param width the first layer's filters
     * @param seed the initialization seed
     * @return the discriminator
     */
    public static PatchDiscriminator of(int channels, int width, long seed) {
        return new PatchDiscriminator(channels, width, seed);
    }

    private static Var leaky(Var x) {
        return x.relu().sub(x.neg().relu().mul(0.2f));
    }

    @Override
    public Var forward(Var images) {
        Var h = Functional.conv(images, weights.get(0), biases.get(0), new int[] {2, 2}, new int[] {1, 1},
                new int[] {1, 1}, new int[] {1, 1}, 1);
        h = leaky(h);
        h = leaky(Functional.conv(h, weights.get(1), biases.get(1), new int[] {1, 1}, new int[] {1, 1},
                new int[] {1, 1}, new int[] {1, 1}, 1));
        return Functional.conv(h, weights.get(2), biases.get(2), new int[] {1, 1}, new int[] {1, 1},
                new int[] {1, 1}, new int[] {1, 1}, 1);
    }

    /**
     * The mean patch logit of each image, the single score core {@code Adversarial} expects.
     *
     * @param images {@code [n, channels, h, w]}
     * @return {@code [n, 1]}
     */
    public Var score(Var images) {
        Var p = forward(images);
        int n = p.shape().dim(0);
        return p.reshape(n, Math.toIntExact(p.shape().size() / n)).mean(1, true);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < weights.size(); i++) {
            p.add(weights.get(i));
            p.add(biases.get(i));
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), 1, (input.dim(2) + 1) / 2, (input.dim(3) + 1) / 2);
    }
}
