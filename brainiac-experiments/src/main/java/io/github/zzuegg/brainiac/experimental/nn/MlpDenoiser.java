package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The smallest useful {@link Denoiser}: an MLP over the flattened sample, sinusoidal
 * features of the time and a one-hot class (with an extra "no label" slot for
 * classifier-free guidance), with a residual connection around each hidden layer. Good
 * enough for 2-D toy distributions and 8x8 images.
 *
 * <pre>{@code
 * Denoiser net = MlpDenoiser.of(64, 4, 128, 2, 42);   // 8x8 images, 4 classes
 * Var eps = net.forward(Var.of(noisy), times, labels);
 * }</pre>
 */
public final class MlpDenoiser implements Denoiser {
    private static final int TIME_FEATURES = 16;
    private final int dimension;
    private final int classes;
    private final Dense in;
    private final List<Dense> hidden = new ArrayList<>();
    private final Dense out;

    private MlpDenoiser(int dimension, int classes, int width, int depth, long seed) {
        if (dimension < 1 || classes < 0 || width < 1 || depth < 0) {
            throw new IllegalArgumentException("bad MlpDenoiser size");
        }
        SplittableRandom random = new SplittableRandom(seed);
        this.dimension = dimension;
        this.classes = classes;
        in = GenBlocks.linear(dimension + TIME_FEATURES + classes + 1, width, random);
        for (int i = 0; i < depth; i++) {
            hidden.add(GenBlocks.linear(width, width, random));
        }
        out = GenBlocks.linear(width, dimension, random);
    }

    /**
     * Creates the network.
     *
     * @param dimension the sample size
     * @param classes the number of classes, 0 for unconditional only
     * @param width the hidden width
     * @param depth the number of residual hidden layers
     * @param seed the initialization seed
     * @return the network
     */
    public static MlpDenoiser of(int dimension, int classes, int width, int depth, long seed) {
        return new MlpDenoiser(dimension, classes, width, depth, seed);
    }

    @Override
    public Var forward(Var x, float[] t, int[] labels) {
        Tensor cond = Tensor.concat(1, GenBlocks.timeEmbedding(t, TIME_FEATURES), GenBlocks.oneHot(labels, classes))
                .to(x.value().device());
        Var h = GenBlocks.silu(in.forward(Var.concat(1, x, Var.of(cond))));
        for (Dense l : hidden) {
            h = h.add(GenBlocks.silu(l.forward(h)));
        }
        return out.forward(h);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(in.parameters());
        for (Dense l : hidden) {
            p.addAll(l.parameters());
        }
        p.addAll(out.parameters());
        return p;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int classes() {
        return classes;
    }
}
