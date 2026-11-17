package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A small PixelCNN for binary images {@code [batch, 1, h, w]}: one {@link MaskedConv2d} of
 * type A, residual blocks of type-B masked convolutions, and a 1x1 output to one Bernoulli
 * logit per pixel. {@link #forward} returns the logits of every pixel given the pixels before
 * it in raster order, so the exact negative log-likelihood ({@link #nll}) is one pass;
 * {@link #sample} generates pixel by pixel.
 *
 * <p>Paper: van den Oord et al., "Conditional Image Generation with PixelCNN Decoders" (2016),
 * <a href="https://arxiv.org/abs/1606.05328">arXiv:1606.05328</a>.
 */
public final class PixelCnn implements Layer {

    private final MaskedConv2d first;
    private final List<MaskedConv2d> blocks = new ArrayList<>();
    private final MaskedConv2d out;

    /**
     * Creates the network.
     *
     * @param hidden feature maps per layer
     * @param blocks residual type-B blocks
     * @param random the random source
     */
    public PixelCnn(int hidden, int blocks, RandomGenerator random) {
        first = new MaskedConv2d(1, hidden, 5, true, random);
        for (int i = 0; i < blocks; i++) {
            this.blocks.add(new MaskedConv2d(hidden, hidden, 3, false, random));
        }
        out = new MaskedConv2d(hidden, 1, 1, false, random);
    }

    @Override
    public Var forward(Var input) {
        Var h = first.forward(input).relu();
        for (MaskedConv2d b : blocks) {
            h = h.add(b.forward(h).relu());
        }
        return out.forward(h);
    }

    /**
     * The negative log-likelihood in nats per image.
     *
     * @param images binary images {@code [batch, 1, h, w]}
     * @return the mean NLL
     */
    public Var nll(Tensor images) {
        Var logits = forward(Var.of(images));
        int pixels = Math.toIntExact(images.size() / images.shape().dim(0));
        Var p = logits.sigmoid();
        return Loss.BINARY_CROSS_ENTROPY.compute(p, images).mul(pixels);
    }

    /**
     * Generates images pixel by pixel.
     *
     * @param n the number of images
     * @param height image height
     * @param width image width
     * @param random the sampling source
     * @return binary images {@code [n, 1, height, width]}
     */
    public Tensor sample(int n, int height, int width, RandomGenerator random) {
        float[] img = new float[n * height * width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float[] logits = forward(Var.of(Tensor.of(Shape.of(n, 1, height, width), img))).value().toArray();
                for (int s = 0; s < n; s++) {
                    int i = (s * height + y) * width + x;
                    double p = 1 / (1 + Math.exp(-logits[i]));
                    img[i] = random.nextDouble() < p ? 1f : 0f;
                }
            }
        }
        return Tensor.of(Shape.of(n, 1, height, width), img);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(first.parameters());
        for (MaskedConv2d b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(out.parameters());
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "PixelCnn(" + blocks.size() + " blocks)";
    }
}
