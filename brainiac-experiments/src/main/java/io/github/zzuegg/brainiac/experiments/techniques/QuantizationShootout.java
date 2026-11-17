package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.Awq;
import io.github.zzuegg.brainiac.experimental.nn.Fp8Format;
import io.github.zzuegg.brainiac.experimental.nn.Gptq;
import io.github.zzuegg.brainiac.experimental.nn.LsqDense;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Trains a small MLP classifier on flattened {@link ShapeImages}, then quantizes every weight
 * matrix to {@code bits} bits with each post-training method and reports test accuracy:
 * round-to-nearest, {@link Gptq} (with and without act-order), {@link Awq}, and per-tensor
 * {@link Fp8Format} E4M3 and E5M2; plus {@link LsqDense} quantization-aware fine-tuning, which
 * trains the low-bit weights and their step sizes. GPTQ and AWQ calibrate on 256 training
 * images, with each layer's inputs taken from the already-quantized layers before it.
 */
public final class QuantizationShootout {

    private static final ShapeImages IMAGES = new ShapeImages(8, 0.1f);

    private QuantizationShootout() {
    }

    /**
     * Runs the comparison.
     *
     * @param bits the integer bit width for RTN, GPTQ, AWQ and LSQ
     * @param seed the seed
     * @return method name to test accuracy, starting with {@code "float"}
     */
    public static Map<String, Float> run(int bits, long seed) {
        Model model = Model.sequential().input(64).dense(64, Activation.RELU).dense(64, Activation.RELU)
                .dense(ShapeImages.CLASSES, Activation.IDENTITY).seed(seed).build();
        SplittableRandom rnd = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 300; s++) {
            Tensor[] b = IMAGES.generate(64, rnd.nextLong());
            Functional.crossEntropy(model.forward(flat(b[0])), b[1]).backward();
            opt.step(model.parameters());
        }
        Tensor[] test = IMAGES.generate(512, seed + 1);
        Tensor calibration = flat(IMAGES.generate(256, seed + 2)[0]).value();
        List<Dense> dense = new ArrayList<>();
        for (Layer l : model.layers()) {
            if (l instanceof Dense d) {
                dense.add(d);
            }
        }
        List<Tensor> original = new ArrayList<>();
        for (Dense d : dense) {
            original.add(d.weight().value());
        }
        Map<String, Float> out = new LinkedHashMap<>();
        out.put("float", accuracy(model, test));
        for (String method : List.of("rtn", "gptq", "gptq-act-order", "awq", "fp8-e4m3", "fp8-e5m2")) {
            Tensor x = calibration;
            for (int i = 0; i < dense.size(); i++) {
                Tensor w = original.get(i);
                Tensor q = switch (method) {
                    case "rtn" -> Gptq.roundToNearest(w, bits);
                    case "gptq" -> new Gptq(bits).quantize(w, x);
                    case "gptq-act-order" -> new Gptq(bits).actOrder(true).quantize(w, x);
                    case "awq" -> Awq.quantize(w, x, bits).weight();
                    case "fp8-e4m3" -> Fp8Format.E4M3.quantize(w);
                    default -> Fp8Format.E5M2.quantize(w);
                };
                dense.get(i).weight().assign(q);
                x = dense.get(i).forward(Var.of(x)).value();   // the next layer sees quantized inputs
            }
            out.put(method, accuracy(model, test));
            for (int i = 0; i < dense.size(); i++) {
                dense.get(i).weight().assign(original.get(i));
            }
        }
        out.put("lsq-qat", lsq(dense, bits, rnd, test));
        return out;
    }

    private static float lsq(List<Dense> dense, int bits, SplittableRandom rnd, Tensor[] test) {
        List<LsqDense> layers = new ArrayList<>();
        List<Var> params = new ArrayList<>();
        for (Dense d : dense) {
            LsqDense l = LsqDense.from(d, bits, 0);
            layers.add(l);
            params.addAll(l.parameters());
        }
        Optimizer opt = Optimizer.adam(1e-3f);
        for (int s = 0; s < 150; s++) {
            Tensor[] b = IMAGES.generate(64, rnd.nextLong());
            Functional.crossEntropy(forward(layers, flat(b[0])), b[1]).backward();
            opt.step(params);
        }
        return forward(layers, flat(test[0])).value().argmax(1).equalTo(test[1]).mean().item();
    }

    private static Var forward(List<LsqDense> layers, Var x) {
        for (LsqDense l : layers) {
            x = l.forward(x);
        }
        return x;
    }

    private static Var flat(Tensor images) {
        return Var.of(images.reshape(images.shape().dim(0), 64));
    }

    private static float accuracy(Model model, Tensor[] test) {
        return model.forward(flat(test[0])).value().argmax(1).equalTo(test[1]).mean().item();
    }
}
