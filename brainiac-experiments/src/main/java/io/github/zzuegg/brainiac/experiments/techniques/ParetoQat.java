package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.ParetoQDense;
import io.github.zzuegg.brainiac.experimental.train.ParetoQSchedule;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * ParetoQ's training recipe against its two alternatives, on the {@link QuantizationShootout}
 * MLP (flattened {@link ShapeImages}, 64-64-64-classes, every weight matrix quantized): with the
 * same step budget and the same bit-specific {@link ParetoQDense} quantizer,
 * <ul>
 * <li>{@code paretoq}: most of the steps in full precision, the rest quantization-aware
 *     ({@link ParetoQSchedule}; the paper's 10% QAT is tens of millions of tokens, here a
 *     300-step run needs about 30% to let the scales settle);</li>
 * <li>{@code qat}: quantization-aware from the first step;</li>
 * <li>{@code ptq}: all steps in full precision, then the weights quantized once.</li>
 * </ul>
 * Reported as test loss and accuracy, with the float model for reference.
 */
public final class ParetoQat {

    private static final ShapeImages IMAGES = new ShapeImages(8, 0.1f);

    /**
     * One training regime's outcome.
     *
     * @param loss the test cross-entropy
     * @param accuracy the test accuracy
     */
    public record Outcome(float loss, float accuracy) {
    }

    /**
     * All regimes at one bit width.
     *
     * @param bits the bit width
     * @param full the full-precision model
     * @param paretoq 90% float then 10% QAT
     * @param qat QAT from scratch
     * @param ptq post-training quantization
     */
    public record Result(double bits, Outcome full, Outcome paretoq, Outcome qat, Outcome ptq) {
    }

    private ParetoQat() {
    }

    /**
     * Runs the comparison.
     *
     * @param bits 1, 1.58, 2, 3 or 4
     * @param steps training steps per regime
     * @param qatShare the share of the steps the ParetoQ run trains quantized (0.1 in the paper)
     * @param seed the seed
     * @return the result
     */
    public static Result run(double bits, int steps, double qatShare, long seed) {
        Tensor[] test = IMAGES.generate(1024, seed + 1);
        Outcome[] ptq = new Outcome[2];
        Outcome paretoq = train(bits, steps, qatShare, seed, test, null);
        Outcome qat = train(bits, steps, 1.0, seed, test, null);
        train(bits, steps, 0.0, seed, test, ptq);
        return new Result(bits, ptq[0], paretoq, qat, ptq[1]);
    }

    private static Outcome train(double bits, int steps, double qatShare, long seed, Tensor[] test, Outcome[] ptq) {
        SplittableRandom init = new SplittableRandom(seed);
        int[] sizes = {64, 64, 64, ShapeImages.CLASSES};
        List<ParetoQDense> layers = new ArrayList<>();
        List<Var> params = new ArrayList<>();
        for (int i = 0; i + 1 < sizes.length; i++) {
            ParetoQDense l = new ParetoQDense(Init.XAVIER.create(Shape.of(sizes[i], sizes[i + 1]), sizes[i],
                    sizes[i + 1], init), Tensor.zeros(Shape.of(sizes[i + 1])),
                    i + 2 < sizes.length ? Activation.RELU : Activation.IDENTITY, bits);
            layers.add(l);
            params.addAll(l.parameters());
        }
        Optimizer opt = new ParetoQSchedule(Optimizer.adam(3e-3f), layers, ptq != null ? Long.MAX_VALUE : steps,
                qatShare);
        SplittableRandom data = new SplittableRandom(seed + 7);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = IMAGES.generate(64, data.nextLong());
            Functional.crossEntropy(forward(layers, flat(b[0])), b[1]).backward();
            opt.step(params);
        }
        if (ptq != null) {
            ptq[0] = evaluate(layers, test);
            layers.forEach(l -> l.quantize(true));
            ptq[1] = evaluate(layers, test);
            return ptq[1];
        }
        return evaluate(layers, test);
    }

    private static Outcome evaluate(List<ParetoQDense> layers, Tensor[] test) {
        Var logits = forward(layers, flat(test[0]));
        float loss = Functional.crossEntropy(logits, test[1]).value().item();
        float acc = logits.value().argmax(1).equalTo(test[1]).mean().item();
        return new Outcome(loss, acc);
    }

    private static Var forward(List<ParetoQDense> layers, Var x) {
        for (ParetoQDense l : layers) {
            x = l.forward(x);
        }
        return x;
    }

    private static Var flat(Tensor images) {
        return Var.of(images.reshape(images.shape().dim(0), 64));
    }
}
