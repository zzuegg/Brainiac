package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.autograd.Trig;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Maps low-dimensional coordinates to sines and cosines so that an MLP can learn
 * high-frequency functions of them, the trick that makes coordinate networks (NeRF,
 * neural fields, SIREN alternatives) work. Plain MLPs on raw coordinates are biased
 * towards smooth functions ("spectral bias") and blur fine detail.
 *
 * <ul>
 * <li>{@link #positional} is NeRF's encoding (Mildenhall et al.,
 *     <a href="https://arxiv.org/abs/2003.08934">2003.08934</a>): {@code sin(2^k pi x), cos(2^k pi x)}
 *     for {@code k < octaves}, per coordinate.</li>
 * <li>{@link #gaussian} is random Fourier features (Tancik et al.,
 *     <a href="https://arxiv.org/abs/2006.10739">2006.10739</a>): {@code sin(2 pi B x), cos(2 pi B x)}
 *     with a fixed random matrix {@code B ~ N(0, scale^2)}; the scale sets the bandwidth.</li>
 * </ul>
 *
 * <p>Input {@code [batch, d]}, output {@code [batch, 2m]} (plus the raw input when kept). No
 * trainable parameters; gradients flow to the coordinates.
 */
public final class FourierFeatures implements Layer {
    private final Tensor frequencies;   // [d, m], already including 2 pi
    private final boolean keepInput;

    private FourierFeatures(Tensor frequencies, boolean keepInput) {
        this.frequencies = frequencies;
        this.keepInput = keepInput;
    }

    /**
     * NeRF positional encoding.
     *
     * @param inputs the coordinate count d
     * @param octaves the number of frequencies per coordinate
     * @param keepInput whether to append the raw coordinates
     * @return the layer
     */
    public static FourierFeatures positional(int inputs, int octaves, boolean keepInput) {
        float[] f = new float[inputs * inputs * octaves];
        for (int i = 0; i < inputs; i++) {
            for (int k = 0; k < octaves; k++) {
                f[i * inputs * octaves + i * octaves + k] = (float) (Math.PI * (1L << k));
            }
        }
        return new FourierFeatures(Tensor.of(Shape.of(inputs, inputs * octaves), f), keepInput);
    }

    /**
     * Random Gaussian Fourier features.
     *
     * @param inputs the coordinate count d
     * @param features the number of random frequencies m
     * @param scale the standard deviation of the frequencies
     * @param seed the seed
     * @param keepInput whether to append the raw coordinates
     * @return the layer
     */
    public static FourierFeatures gaussian(int inputs, int features, float scale, long seed, boolean keepInput) {
        Tensor b = Tensor.normal(Shape.of(inputs, features), 0, scale, new SplittableRandom(seed))
                .mul((float) (2 * Math.PI));
        return new FourierFeatures(b, keepInput);
    }

    /**
     * Describes a random Fourier feature layer for model builders.
     *
     * @param features the number of random frequencies
     * @param scale their standard deviation
     * @param keepInput whether to append the raw coordinates
     * @return the spec
     */
    public static LayerSpec spec(int features, float scale, boolean keepInput) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), 2 * features + (keepInput ? input.dim(-1) : 0));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return gaussian(input.dim(-1), features, scale, random.nextLong(), keepInput);
            }
        };
    }

    /**
     * Describes a NeRF positional encoding for model builders.
     *
     * @param octaves the number of frequencies per coordinate
     * @param keepInput whether to append the raw coordinates
     * @return the spec
     */
    public static LayerSpec positionalSpec(int octaves, boolean keepInput) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int d = input.dim(-1);
                return Shape.of(input.dim(0), 2 * d * octaves + (keepInput ? d : 0));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return positional(input.dim(-1), octaves, keepInput);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() != 2 || input.shape().dim(1) != frequencies.shape().dim(0)) {
            throw new IllegalArgumentException("FourierFeatures expects [batch, " + frequencies.shape().dim(0)
                    + "], got " + input.shape());
        }
        Var phase = input.matmul(Var.of(frequencies.to(input.value().device())));
        Var sin = Trig.sin(phase);
        Var cos = Trig.cos(phase);
        return keepInput ? Var.concat(1, sin, cos, input) : Var.concat(1, sin, cos);
    }

    @Override
    public int outputSize(int inputSize) {
        return 2 * frequencies.shape().dim(1) + (keepInput ? inputSize : 0);
    }
}
