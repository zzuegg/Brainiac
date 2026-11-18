package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Fine-tuning that keeps a pretrained model's robustness to distribution shift:
 *
 * <ul>
 *   <li>LP-FT ({@link #linearProbe} then {@link #fineTune}): fit a linear probe on frozen
 *       features first, then fine-tune everything from that head, so a random head does not
 *       distort the features early on (Kumar et al., ICLR 2022,
 *       <a href="https://arxiv.org/abs/2202.10054">arXiv:2202.10054</a>).</li>
 *   <li>WiSE-FT ({@link #interpolate}): the weights {@code (1 - a) theta_0 + a theta_ft}
 *       between the zero-shot or probed model and the fine-tuned one, which keeps most of the
 *       in-distribution gain and recovers robustness (Wortsman et al., CVPR 2022,
 *       <a href="https://arxiv.org/abs/2109.01903">arXiv:2109.01903</a>).</li>
 * </ul>
 */
public final class RobustFineTuning {

    private RobustFineTuning() {
    }

    /**
     * Trains only the head on frozen features.
     *
     * @param model the classifier; only its head changes
     * @param x training inputs
     * @param y training labels
     * @param steps Adam steps
     * @param learningRate the step size
     * @param seed the batch-sampling seed
     */
    public static void linearProbe(FeatureClassifier model, Tensor x, int[] y, int steps, float learningRate,
            long seed) {
        Tensor features = model.features(x);
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(features, y, 64, r);
            Functional.crossEntropy(model.head().forward(Var.of(b[0])), b[1]).backward();
            opt.step(model.head().parameters());
        }
    }

    /**
     * Fine-tunes all parameters.
     *
     * @param model the classifier
     * @param x training inputs
     * @param y training labels
     * @param steps Adam steps
     * @param learningRate the step size
     * @param seed the batch-sampling seed
     */
    public static void fineTune(FeatureClassifier model, Tensor x, int[] y, int steps, float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(x, y, 64, r);
            Functional.crossEntropy(model.logits(Var.of(b[0]), true), b[1]).backward();
            opt.step(model.parameters());
        }
    }

    /**
     * WiSE-FT weight interpolation.
     *
     * @param start the zero-shot or probed weights
     * @param fineTuned the fine-tuned weights
     * @param alpha the mixing weight of the fine-tuned model (0 = start, 1 = fine-tuned)
     * @return {@code (1 - alpha) start + alpha fineTuned}
     */
    public static List<Tensor> interpolate(List<Tensor> start, List<Tensor> fineTuned, double alpha) {
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < start.size(); i++) {
            out.add(start.get(i).mul((float) (1 - alpha)).add(fineTuned.get(i).mul((float) alpha)));
        }
        return out;
    }

    private static Tensor[] batch(Tensor x, int[] y, int size, SplittableRandom r) {
        float[] rows = new float[size];
        float[] ys = new float[size];
        for (int i = 0; i < size; i++) {
            int k = r.nextInt(y.length);
            rows[i] = k;
            ys[i] = y[k];
        }
        return new Tensor[] {x.gather(0, Tensor.of(Shape.of(size), rows)), Tensor.of(Shape.of(size), ys)};
    }
}
