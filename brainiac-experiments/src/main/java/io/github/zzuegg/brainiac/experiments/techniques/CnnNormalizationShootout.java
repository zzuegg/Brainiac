package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.BatchRenorm;
import io.github.zzuegg.brainiac.experimental.nn.FilterResponseNorm;
import io.github.zzuegg.brainiac.experimental.nn.WeightNormDense;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.GlobalPool;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Wave 6 item 2.3, the CNN arm of {@link NormalizationShootout} (which is transformer only):
 * a two-convolution CNN on {@link ShapeImages} with each normaliser after both convolutions,
 * trained at batch sizes 2, 8 and 64 and, optionally, on non-iid batches (each batch holds
 * only two classes, as when data arrive sorted). Compared: core {@code BatchNorm} (the
 * baseline), core {@code GroupNorm}, {@link BatchRenorm} and {@link FilterResponseNorm}
 * (with its TLU in place of the ReLU), plus no normaliser. A second, MLP arm compares a core
 * dense hidden layer with {@link WeightNormDense}.
 */
public final class CnnNormalizationShootout {

    /** The normalisers compared. */
    public enum Norm {
        /** no normalisation */
        NONE,
        /** core batch norm */
        BATCH_NORM,
        /** core group norm, 4 groups */
        GROUP_NORM,
        /** batch renormalisation */
        BATCH_RENORM,
        /** filter response norm + TLU */
        FRN
    }

    private static final ShapeImages IMAGES = new ShapeImages(8, 0.1f);

    private CnnNormalizationShootout() {
    }

    private static void normAndActivation(Model.Builder b, Norm n) {
        switch (n) {
            case NONE -> b.activation(Activation.RELU);
            case BATCH_NORM -> b.batchNorm().activation(Activation.RELU);
            case GROUP_NORM -> b.groupNorm(4).activation(Activation.RELU);
            case BATCH_RENORM -> b.add(BatchRenorm.spec(100)).activation(Activation.RELU);
            case FRN -> b.add(FilterResponseNorm.spec());
        }
    }

    /**
     * Builds the CNN.
     *
     * @param n the normaliser
     * @param seed the weight seed
     * @return the model
     */
    public static Model build(Norm n, long seed) {
        Model.Builder b = Model.sequential().input(1, 8, 8).seed(seed).conv2d(16, 3, Activation.IDENTITY);
        normAndActivation(b, n);
        b.conv2d(32, 3, Activation.IDENTITY);
        normAndActivation(b, n);
        return b.layer(GlobalPool.avg()).dense(ShapeImages.CLASSES, Activation.IDENTITY).build();
    }

    /** Draws a batch; non-iid batches hold two classes only. */
    private static Tensor[] batch(int size, boolean iid, SplittableRandom r) {
        if (iid) {
            return IMAGES.generate(size, r.nextLong());
        }
        int a = r.nextInt(ShapeImages.CLASSES);
        int c = (a + 1 + r.nextInt(ShapeImages.CLASSES - 1)) % ShapeImages.CLASSES;
        Tensor pool = null;
        Tensor ids = null;
        float[] img = new float[size * 64];
        float[] lab = new float[size];
        int got = 0;
        while (got < size) {
            Tensor[] g = IMAGES.generate(32, r.nextLong());
            pool = g[0];
            ids = g[1];
            float[] pv = pool.toArray();
            float[] iv = ids.toArray();
            for (int i = 0; i < 32 && got < size; i++) {
                if ((int) iv[i] == a || (int) iv[i] == c) {
                    System.arraycopy(pv, i * 64, img, got * 64, 64);
                    lab[got++] = iv[i];
                }
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(size, 1, 8, 8), img), Tensor.of(Shape.of(size), lab)};
    }

    /**
     * Trains the CNN with one normaliser.
     *
     * @param n the normaliser
     * @param batch images per step
     * @param iid whether batches are iid (else two classes per batch)
     * @param steps Adam steps
     * @param seed the seed
     * @return accuracy on 512 test images, in inference mode (running statistics)
     */
    public static float run(Norm n, int batch, boolean iid, int steps, long seed) {
        Model model = build(n, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = batch(batch, iid, r);
            Functional.crossEntropy(model.forward(Var.of(d[0]), true), d[1]).backward();
            opt.step(model.parameters());
        }
        Tensor[] test = IMAGES.generate(512, seed + 99);
        return model.forward(Var.of(test[0])).value().argmax(1).equalTo(test[1]).mean().item();
    }

    /**
     * The MLP arm: two hidden layers of 64 units on flattened images, plain dense or weight-normalised.
     *
     * @param weightNorm whether the hidden layers use {@link WeightNormDense}
     * @param batch images per step
     * @param steps SGD-with-momentum steps
     * @param seed the seed
     * @return {@code {mean training loss over the last 20 steps, test accuracy}}
     */
    public static float[] mlp(boolean weightNorm, int batch, int steps, long seed) {
        Model.Builder b = Model.sequential().input(64).seed(seed);
        for (int i = 0; i < 2; i++) {
            if (weightNorm) {
                b.add(WeightNormDense.spec(64, Activation.RELU));
            } else {
                b.dense(64, Activation.RELU);
            }
        }
        Model model = b.dense(ShapeImages.CLASSES, Activation.IDENTITY).build();
        Optimizer opt = Optimizer.momentum(0.05f, 0.9f);
        SplittableRandom r = new SplittableRandom(seed);
        double tail = 0;
        for (int s = 0; s < steps; s++) {
            Tensor[] d = IMAGES.generate(batch, r.nextLong());
            Var loss = Functional.crossEntropy(model.forward(Var.of(d[0].reshape(batch, 64)), true), d[1]);
            tail += s >= steps - 20 ? loss.value().item() / 20 : 0;
            loss.backward();
            opt.step(model.parameters());
        }
        Tensor[] test = IMAGES.generate(512, seed + 99);
        float acc = model.forward(Var.of(test[0].reshape(512, 64))).value().argmax(1).equalTo(test[1]).mean().item();
        return new float[] {(float) tail, acc};
    }
}
