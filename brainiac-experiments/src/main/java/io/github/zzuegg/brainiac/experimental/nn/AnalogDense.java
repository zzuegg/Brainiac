package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A dense layer as it would run on an analog in-memory computing crossbar, with
 * hardware-aware training (Joshi et al., <a href="https://arxiv.org/abs/1906.03138">1906.03138</a>;
 * IBM aihwkit, <a href="https://arxiv.org/abs/2104.02184">2104.02184</a>). Weights are
 * conductances limited to {@code [-clip, clip]}; programming them onto a chip adds Gaussian
 * noise of {@code noise * clip}; the column currents are read by an ADC of a given bit width
 * over {@code [-range, range]}.
 *
 * <p>While training the layer injects fresh weight noise every pass and quantizes outputs
 * with a straight-through estimator, so the network learns weights that tolerate the chip.
 * {@link #program} samples one fixed noisy chip instance that inference then uses until
 * {@link #unprogram}; this is how robustness is measured.
 */
public final class AnalogDense implements Layer {
    private final int inputs;
    private final int units;
    private final float clip;
    private final float trainingNoise;
    private final int adcBits;
    private final float adcRange;
    private final Var weight;
    private final Var bias;
    private final SplittableRandom random;
    private Tensor programmed;

    private AnalogDense(int inputs, int units, float clip, float trainingNoise, int adcBits, float adcRange,
            SplittableRandom random) {
        this.inputs = inputs;
        this.units = units;
        this.clip = clip;
        this.trainingNoise = trainingNoise;
        this.adcBits = adcBits;
        this.adcRange = adcRange;
        this.weight = Var.param(Init.XAVIER.create(Shape.of(inputs, units), inputs, units, random));
        this.bias = Var.param(Tensor.zeros(Shape.of(units)));
        this.random = random.split();
    }

    /**
     * Creates a layer.
     *
     * @param inputs the input width
     * @param units the output width
     * @param clip the largest weight magnitude a device holds
     * @param trainingNoise the weight noise injected while training, relative to {@code clip}; 0 for none
     * @param adcBits the ADC resolution, 0 for an ideal readout
     * @param adcRange the ADC full scale
     * @param seed the seed
     * @return the layer
     */
    public static AnalogDense of(int inputs, int units, float clip, float trainingNoise, int adcBits, float adcRange,
            long seed) {
        return new AnalogDense(inputs, units, clip, trainingNoise, adcBits, adcRange, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders.
     *
     * @param units the output width
     * @param clip the weight limit
     * @param trainingNoise the training weight noise relative to {@code clip}
     * @param adcBits the ADC resolution, 0 for none
     * @param adcRange the ADC full scale
     * @return the spec
     */
    public static LayerSpec spec(int units, float clip, float trainingNoise, int adcBits, float adcRange) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(units);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new AnalogDense(input.dim(-1), units, clip, trainingNoise, adcBits, adcRange, random);
            }
        };
    }

    /**
     * Writes the weights to a simulated chip: clipped, plus a fixed Gaussian programming error.
     *
     * @param noise the programming noise relative to {@code clip}
     * @param chip the random source for this chip instance
     */
    public void program(float noise, SplittableRandom chip) {
        Tensor w = weight.value().maximum(Tensor.scalar(-clip)).minimum(Tensor.scalar(clip));
        programmed = w.add(Tensor.normal(w.shape(), 0, noise * clip, chip));
    }

    /** Returns to ideal (noise-free, clipped) inference. */
    public void unprogram() {
        programmed = null;
    }

    private Var adc(Var y) {
        if (adcBits <= 0) {
            return y;
        }
        float levels = (1 << (adcBits - 1)) - 1;
        Var clamped = y.clamp(-adcRange, adcRange);
        Tensor step = clamped.value().mul(levels / adcRange).round().mul(adcRange / levels);
        return clamped.add(Var.of(step.sub(clamped.value())));   // straight-through rounding
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().dim(-1) != inputs) {
            throw new IllegalArgumentException("AnalogDense expects [*, " + inputs + "], got " + input.shape());
        }
        Var w;
        if (programmed != null && !training) {
            w = Var.of(programmed);
        } else {
            w = weight.clamp(-clip, clip);
            if (training && trainingNoise > 0) {
                w = w.add(Tensor.normal(w.shape(), 0, trainingNoise * clip, random));
            }
        }
        return adc(input.matmul(w)).add(bias);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }
}
