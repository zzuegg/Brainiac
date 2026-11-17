package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.ZLoss;
import io.github.zzuegg.brainiac.experimental.train.DynamicLossScaler;
import io.github.zzuegg.brainiac.experimental.train.GradientNoise;
import io.github.zzuegg.brainiac.experimental.train.RDrop;
import io.github.zzuegg.brainiac.experimental.train.TrainingSchedules;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.LearningRateSchedule;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Small experiments with training tricks, each on a task where its effect shows:
 *
 * <ul>
 *   <li>{@link #schedules}: constant, cosine, warmup-stable-decay, 1cycle and cosine with
 *       restarts at the same peak rate on a noisy regression; decaying schedules end lower
 *       because the final small steps average out gradient noise.</li>
 *   <li>{@link #zLoss}: a classifier trained with and without z-loss; z-loss keeps the
 *       softmax normalizer {@code log Z} near zero.</li>
 *   <li>{@link #rDrop}: dropout classifiers on few noisy shape images, with and without
 *       R-Drop's consistency term.</li>
 *   <li>{@link #gradientNoise}: a deep, narrow MLP with and without annealed gradient noise.</li>
 *   <li>{@link #lossScaling}: float16 training whose gradients underflow to zero without loss
 *       scaling and train normally with {@link DynamicLossScaler}.</li>
 * </ul>
 */
public final class TrainingTricksExperiment {

    private TrainingTricksExperiment() {
    }

    /**
     * Trains the same regression under each schedule.
     *
     * @param steps training steps
     * @param seed the seed
     * @return schedule name to the mean squared error on fresh data
     */
    public static Map<String, Float> schedules(int steps, long seed) {
        Map<String, LearningRateSchedule> schedules = new LinkedHashMap<>();
        schedules.put("constant", (s, base) -> base);
        schedules.put("cosine", LearningRateSchedule.cosine(steps, 0f));
        schedules.put("wsd", TrainingSchedules.warmupStableDecay(steps / 20, steps * 7 / 10, steps / 4, 0f));
        schedules.put("1cycle", TrainingSchedules.oneCycle(steps));
        schedules.put("sgdr", TrainingSchedules.cosineWithRestarts(steps / 7, 2f, 0f));
        Map<String, Float> out = new LinkedHashMap<>();
        for (Map.Entry<String, LearningRateSchedule> e : schedules.entrySet()) {
            Model m = Model.sequential().input(8).dense(32, Activation.TANH).dense(1, Activation.IDENTITY).seed(seed)
                    .build();
            SplittableRandom r = new SplittableRandom(seed);
            Tensor teacher = Tensor.normal(Shape.of(8, 1), 0, 1, r);
            Optimizer opt = Optimizer.adam(0.03f);
            for (int s = 1; s <= steps; s++) {
                opt.learningRate(Math.max(1e-6f, e.getValue().rate(s, 0.03f)));
                Tensor x = Tensor.normal(Shape.of(16, 8), 0, 1, r);
                Tensor y = x.matmul(teacher).tanh().add(Tensor.normal(Shape.of(16, 1), 0, 0.3f, r));
                m.forward(Var.of(x)).sub(y).square().mean().backward();
                opt.step(m.parameters());
            }
            Tensor x = Tensor.normal(Shape.of(1024, 8), 0, 1, new SplittableRandom(seed + 1));
            out.put(e.getKey(), m.forward(Var.of(x)).sub(x.matmul(teacher).tanh()).square().mean().value().item());
        }
        return out;
    }

    /**
     * Trains a shape classifier from raw logits with and without z-loss.
     *
     * @param weight the z-loss weight
     * @param seed the seed
     * @return {@code {test accuracy, mean |log Z| on test data}}
     */
    public static float[] zLoss(float weight, long seed) {
        ShapeImages images = new ShapeImages(8, 0.1f);
        Model m = Model.sequential().input(64).dense(64, Activation.RELU).dense(ShapeImages.CLASSES,
                Activation.IDENTITY).seed(seed).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        ZLoss loss = new ZLoss(weight);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < 200; s++) {
            Tensor[] b = images.generate(64, r.nextLong());
            loss.compute(m.forward(Var.of(b[0].reshape(64, 64))), b[1]).backward();
            opt.step(m.parameters());
        }
        Tensor[] test = images.generate(512, seed + 1);
        Var logits = m.forward(Var.of(test[0].reshape(512, 64)));
        float acc = logits.value().argmax(1).equalTo(test[1]).mean().item();
        float logZ = ZLoss.logSumExp(logits).value().abs().mean().item();
        return new float[] {acc, logZ};
    }

    /**
     * Trains a dropout MLP on 96 noisy shape images, with or without R-Drop.
     *
     * @param alpha the R-Drop weight; 0 for plain dropout training
     * @param seed the seed
     * @return the test accuracy
     */
    public static float rDrop(float alpha, long seed) {
        ShapeImages images = new ShapeImages(8, 0.15f);
        Tensor[] train = images.generate(96, seed);
        Model m = Model.sequential().input(64).dense(128, Activation.RELU).dropout(0.3f)
                .dense(ShapeImages.CLASSES, Activation.IDENTITY).seed(seed).build();
        Optimizer opt = Optimizer.adam(2e-3f);
        Var x = Var.of(train[0].reshape(96, 64));
        for (int s = 0; s < 150; s++) {
            Var loss = alpha > 0 ? RDrop.loss(v -> m.forward(v, true), x, train[1], alpha)
                    : Functional.crossEntropy(m.forward(x, true), train[1]);
            loss.backward();
            opt.step(m.parameters());
        }
        Tensor[] test = images.generate(512, seed + 1);
        return m.forward(Var.of(test[0].reshape(512, 64))).value().argmax(1).equalTo(test[1]).mean().item();
    }

    /**
     * Trains a 12-layer, 16-wide tanh MLP on a regression, with or without gradient noise.
     *
     * @param eta the noise variance ({@code 0} for none)
     * @param seed the seed
     * @return the final training loss
     */
    public static float gradientNoise(float eta, long seed) {
        Model.Builder b = Model.sequential().input(4).seed(seed);
        for (int i = 0; i < 12; i++) {
            b.dense(16, Activation.TANH);
        }
        Model m = b.dense(1, Activation.IDENTITY).build();
        SplittableRandom r = new SplittableRandom(seed);
        Tensor x = Tensor.normal(Shape.of(64, 4), 0, 1, r);
        Tensor y = x.slice(1, 0, 1).mul(x.slice(1, 1, 2)).sign();
        Optimizer opt = eta > 0 ? new GradientNoise(Optimizer.adam(3e-3f), eta, 0.55f, seed) : Optimizer.adam(3e-3f);
        float last = 0;
        for (int s = 0; s < 300; s++) {
            Var loss = m.forward(Var.of(x)).sub(y).square().mean();
            last = loss.value().item();
            loss.backward();
            opt.step(m.parameters());
        }
        return last;
    }

    /**
     * Trains a linear regression whose loss is tiny ({@code 1e-7} times the MSE, as the
     * per-token losses of a large batch can be), computing in float16 with float32 master
     * weights.
     *
     * @param scaled whether to use dynamic loss scaling
     * @param seed the seed
     * @return {@code {initial MSE, final MSE}}
     */
    public static float[] lossScaling(boolean scaled, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Tensor teacher = Tensor.normal(Shape.of(8, 1), 0, 1, r);
        Var w = Var.param(Tensor.zeros(Shape.of(8, 1)));
        DynamicLossScaler scaler = new DynamicLossScaler(65536f, 50);
        Optimizer opt = Optimizer.adam(0.05f);
        Tensor xt = Tensor.normal(Shape.of(256, 8), 0, 1, new SplittableRandom(seed + 1));
        float first = mse(w, xt, teacher);
        for (int s = 0; s < 150; s++) {
            Tensor x = Tensor.normal(Shape.of(32, 8), 0, 1, r);
            Tensor y = x.matmul(teacher);
            Var pred = Var.of(x.to(DType.FLOAT16)).matmul(w.to(DType.FLOAT16));
            Var loss = pred.sub(Var.of(y.to(DType.FLOAT16))).square().mean().mul(1e-7f);
            if (scaled) {
                scaler.scale(loss).backward();
                if (scaler.unscale(List.of(w))) {
                    opt.step(List.of(w));
                }
            } else {
                loss.backward();
                opt.step(List.of(w));
            }
            w.zeroGrad();
        }
        return new float[] {first, mse(w, xt, teacher)};
    }

    private static float mse(Var w, Tensor x, Tensor teacher) {
        return x.matmul(w.value()).sub(x.matmul(teacher)).square().mean().item();
    }
}
