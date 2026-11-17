package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.ConvNeXtBlock;
import io.github.zzuegg.brainiac.experimental.nn.DenseBlock2d;
import io.github.zzuegg.brainiac.experimental.nn.DropBlock2d;
import io.github.zzuegg.brainiac.experimental.nn.MbConvBlock;
import io.github.zzuegg.brainiac.experimental.nn.ShakeResidualBlock;
import io.github.zzuegg.brainiac.experimental.nn.SqueezeExcitation;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.GlobalPool;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Wave 6 items 2.2, 2.4 and 2.5: classic CNN regularisers and architectures as rows next to
 * {@link VisionShootout}, trained on a <em>small</em> fixed set of {@link ShapeImages} (the
 * regime where regularisers matter) and compared by test accuracy, parameters and
 * multiply-adds:
 *
 * <ul>
 *   <li>regularisers of the plain CNN: core spatial dropout, {@link DropBlock2d};</li>
 *   <li>residual nets: a plain residual stack, {@link ShakeResidualBlock} Shake-Shake and
 *       ShakeDrop, and ConvNeXt with and without stochastic depth ({@code DropPath});</li>
 *   <li>architectures: {@link DenseBlock2d} (DenseNet), the CNN with
 *       {@link SqueezeExcitation}, and {@link MbConvBlock} (EfficientNet-lite).</li>
 * </ul>
 *
 * <p>{@link #compoundScaling} runs EfficientNet's compound scaling (depth x 1.2, width x 1.1,
 * resolution x 1.15 per step) against width-only scaling at about the same multiply-adds.
 */
public final class ClassicCnnShootout {

    /** The compared models. */
    public enum Architecture {
        /** two convolutions and global pooling, the baseline */
        CNN,
        /** the CNN with core spatial dropout after each convolution */
        CNN_SPATIAL_DROPOUT,
        /** the CNN with DropBlock after each convolution */
        CNN_DROPBLOCK,
        /** the CNN with squeeze-excitation after the second convolution */
        CNN_SE,
        /** a stem and two plain residual blocks */
        RESNET,
        /** a stem and two Shake-Shake blocks */
        SHAKE_SHAKE,
        /** a stem and two ShakeDrop blocks (keep 0.75 and 0.5, the paper's linear decay) */
        SHAKE_DROP,
        /** a stem and two ConvNeXt blocks */
        CONVNEXT,
        /** ConvNeXt with stochastic depth 0.2 */
        CONVNEXT_DROPPATH,
        /** a stem and a DenseNet block */
        DENSENET,
        /** a stem and two MBConv blocks */
        MBCONV
    }

    /**
     * A result.
     *
     * @param accuracy test accuracy
     * @param parameters the parameter count
     * @param multiplyAdds the multiply-adds per image
     */
    public record Result(float accuracy, long parameters, long multiplyAdds) {
    }

    private ClassicCnnShootout() {
    }

    private static LayerSpec convNeXt(float dropPath) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new ConvNeXtBlock(input.dim(0), 3, 4, 1e-6f, dropPath, random);
            }
        };
    }

    /**
     * Builds one model for {@code [1, 8, 8]} images.
     *
     * @param a the architecture
     * @param seed the weight seed
     * @return the model
     */
    public static Model build(Architecture a, long seed) {
        Model.Builder b = Model.sequential().input(1, 8, 8).seed(seed);
        switch (a) {
            case CNN -> b.conv2d(16, 3, Activation.RELU).conv2d(32, 3, Activation.RELU);
            case CNN_SPATIAL_DROPOUT -> b.conv2d(16, 3, Activation.RELU).spatialDropout(0.15f)
                    .conv2d(32, 3, Activation.RELU).spatialDropout(0.15f);
            case CNN_DROPBLOCK -> b.conv2d(16, 3, Activation.RELU).add(DropBlock2d.spec(0.15f, 3))
                    .conv2d(32, 3, Activation.RELU).add(DropBlock2d.spec(0.15f, 3));
            case CNN_SE -> b.conv2d(16, 3, Activation.RELU).conv2d(32, 3, Activation.RELU)
                    .add(SqueezeExcitation.spec(4));
            case RESNET -> b.conv2d(16, 3, Activation.IDENTITY).add(ShakeResidualBlock.spec(ShakeResidualBlock.Mode
                    .SHAKE_DROP, 1f)).add(ShakeResidualBlock.spec(ShakeResidualBlock.Mode.SHAKE_DROP, 1f))
                    .activation(Activation.RELU);
            case SHAKE_SHAKE -> b.conv2d(16, 3, Activation.IDENTITY)
                    .add(ShakeResidualBlock.spec(ShakeResidualBlock.Mode.SHAKE_SHAKE, 1f))
                    .add(ShakeResidualBlock.spec(ShakeResidualBlock.Mode.SHAKE_SHAKE, 1f)).activation(Activation.RELU);
            case SHAKE_DROP -> b.conv2d(16, 3, Activation.IDENTITY)
                    .add(ShakeResidualBlock.spec(ShakeResidualBlock.Mode.SHAKE_DROP, 0.75f))
                    .add(ShakeResidualBlock.spec(ShakeResidualBlock.Mode.SHAKE_DROP, 0.5f)).activation(Activation.RELU);
            case CONVNEXT -> b.conv2d(16, 3, Activation.IDENTITY).add(convNeXt(0f)).add(convNeXt(0f));
            case CONVNEXT_DROPPATH -> b.conv2d(16, 3, Activation.IDENTITY).add(convNeXt(0.2f)).add(convNeXt(0.2f));
            case DENSENET -> b.conv2d(8, 3, Activation.IDENTITY).add(DenseBlock2d.spec(3, 8))
                    .activation(Activation.RELU);
            case MBCONV -> b.conv2d(16, 3, Activation.SILU).add(MbConvBlock.spec(16, 4, 3))
                    .add(MbConvBlock.spec(32, 4, 3));
        }
        return b.layer(GlobalPool.avg()).dense(ShapeImages.CLASSES, Activation.IDENTITY).build();
    }

    /**
     * Estimates a model's multiply-adds per example by running it once and charging each
     * layer, for every weight it holds, one multiply-add per output position: a convolution
     * weight {@code [f, c, k, k]} costs its size times the output's spatial positions, a
     * matrix its size times the output rows. Exact for stride-1 convolutions and dense layers.
     *
     * @param model the model
     * @param input one example's shape
     * @return the estimate
     */
    public static long multiplyAdds(Model model, Shape input) {
        int[] dims = new int[input.rank() + 1];
        dims[0] = 1;
        System.arraycopy(input.toArray(), 0, dims, 1, input.rank());
        long[] total = new long[1];
        model.forward(Var.of(Tensor.zeros(Shape.of(dims))), false, (step, out) -> {
            if (step.layer() == null) {
                return out;
            }
            Shape s = out.shape();
            long positions = s.rank() == 4 ? (long) s.dim(2) * s.dim(3) : s.rank() == 3 ? s.dim(1) : 1;
            for (Var p : step.layer().parameters()) {
                if (p.shape().rank() >= 2) {
                    total[0] += p.shape().size() * positions;
                }
            }
            return out;
        });
        return total[0];
    }

    /**
     * Trains on a fixed small training set and tests on fresh images.
     *
     * @param model the model, for {@code [1, side, side]} inputs
     * @param side the input resolution (images are generated at 8x8 and resized)
     * @param trainImages the size of the fixed training set
     * @param steps Adam steps of 32 images
     * @param seed the seed
     * @return test accuracy on 512 images
     */
    public static float train(Model model, int side, int trainImages, int steps, long seed) {
        ShapeImages images = new ShapeImages(8, 0.2f);
        Tensor[] train = images.generate(trainImages, seed);
        Tensor x = resize(train[0], side);
        float[] labels = train[1].toArray();
        Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = 32;
        for (int s = 0; s < steps; s++) {
            int start = r.nextInt(trainImages - batch + 1);
            Tensor xb = x.slice(0, start, start + batch);
            float[] yb = new float[batch];
            System.arraycopy(labels, start, yb, 0, batch);
            Functional.crossEntropy(model.forward(Var.of(xb), true), Tensor.of(Shape.of(batch), yb)).backward();
            opt.step(model.parameters());
        }
        Tensor[] test = images.generate(512, seed + 99);
        return model.forward(Var.of(resize(test[0], side))).value().argmax(1).equalTo(test[1]).mean().item();
    }

    private static Tensor resize(Tensor images, int side) {
        if (side == images.shape().dim(2)) {
            return images;
        }
        return Functional.resize(Var.of(images), new int[] {side, side}, Functional.Interpolation.LINEAR, false)
                .value();
    }

    /**
     * Trains one architecture on 256 fixed images.
     *
     * @param a the architecture
     * @param steps Adam steps of 32 images
     * @param seed the seed
     * @return the result
     */
    public static Result run(Architecture a, int steps, long seed) {
        Model m = build(a, seed);
        long macs = multiplyAdds(m, Shape.of(1, 8, 8));
        float acc = train(m, 8, 256, steps, seed);
        return new Result(acc, m.parameterCount(), macs);
    }

    /**
     * An EfficientNet-lite family member: a SiLU stem, {@code depth} MBConv blocks of
     * {@code width} channels (expansion 4, SE) and global pooling, on {@code [1, side, side]}.
     *
     * @param width the channels
     * @param depth the MBConv blocks
     * @param side the input resolution
     * @param seed the seed
     * @return the model
     */
    public static Model efficientNet(int width, int depth, int side, long seed) {
        Model.Builder b = Model.sequential().input(1, side, side).seed(seed).conv2d(width, 3, Activation.SILU);
        for (int i = 0; i < depth; i++) {
            b.add(MbConvBlock.spec(width, 4, 3));
        }
        return b.layer(GlobalPool.avg()).dense(ShapeImages.CLASSES, Activation.IDENTITY).build();
    }

    /**
     * Scales the base EfficientNet-lite (width 8, depth 1, resolution 8) by {@code phi} steps,
     * either compound (depth x 1.2, width x 1.1, resolution x 1.15 per step, the paper's
     * coefficients) or width only (x sqrt(2) per step, about the same multiply-adds growth).
     *
     * @param phi the scaling steps
     * @param compound compound scaling if true, width-only otherwise
     * @param steps Adam steps
     * @param seed the seed
     * @return the result
     */
    public static Result compoundScaling(int phi, boolean compound, int steps, long seed) {
        int width;
        int depth;
        int side;
        if (compound) {
            depth = (int) Math.round(1 * Math.pow(1.2, phi) + 0.3);
            width = (int) Math.round(8 * Math.pow(1.1, phi));
            side = (int) Math.round(8 * Math.pow(1.15, phi));
        } else {
            depth = 1;
            width = (int) Math.round(8 * Math.pow(Math.sqrt(2), phi));
            side = 8;
        }
        Model m = efficientNet(width, depth, side, seed);
        long macs = multiplyAdds(m, Shape.of(1, side, side));
        float acc = train(m, side, 256, steps, seed);
        return new Result(acc, m.parameterCount(), macs);
    }
}
