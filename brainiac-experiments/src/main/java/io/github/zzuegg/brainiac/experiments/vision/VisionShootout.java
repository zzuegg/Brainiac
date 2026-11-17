package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.ConvNeXtBlock;
import io.github.zzuegg.brainiac.experimental.nn.DeformableConv2d;
import io.github.zzuegg.brainiac.experimental.nn.MlpMixerBlock;
import io.github.zzuegg.brainiac.experimental.nn.PatchEmbedding;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock;
import io.github.zzuegg.brainiac.experimental.nn.VisionTransformer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.GlobalPool;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.SequencePool;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Trains small image classifiers of each architecture family on {@link ShapeImages} (8x8,
 * six shapes at random positions) with the same budget and compares test accuracy:
 *
 * <ul>
 *   <li>a plain CNN (two 3x3 convolutions and global average pooling), the baseline;</li>
 *   <li>a CNN whose second convolution is a {@link DeformableConv2d};</li>
 *   <li>ConvNeXt: a stem convolution and two {@link ConvNeXtBlock}s;</li>
 *   <li>ViT: {@link VisionTransformer} with learned absolute positions and a class token;</li>
 *   <li>ViT with 2-D RoPE and mean pooling;</li>
 *   <li>MLP-Mixer: {@link PatchEmbedding} and two {@link MlpMixerBlock}s.</li>
 * </ul>
 *
 * <p>Every model is built with {@code Model.sequential()}, so the experimental layers are
 * shown to be drop-ins for core ones.
 */
public final class VisionShootout {

    /** The architectures compared. */
    public enum Architecture {
        /** Two convolutions and global pooling. */
        CNN,
        /** As CNN, with a deformable second convolution. */
        DEFORMABLE_CNN,
        /** Stem convolution and ConvNeXt blocks. */
        CONVNEXT,
        /** Vision Transformer with learned positions. */
        VIT,
        /** Vision Transformer with 2-D rotary positions. */
        VIT_ROPE_2D,
        /** MLP-Mixer. */
        MLP_MIXER
    }

    private static final ShapeImages IMAGES = new ShapeImages(8, 0.1f);

    private VisionShootout() {
    }

    /**
     * Builds one architecture for {@code [1, 8, 8]} images and six classes.
     *
     * @param a the architecture
     * @param seed the weight seed
     * @return the model, which outputs logits
     */
    public static Model build(Architecture a, long seed) {
        Model.Builder b = Model.sequential().input(1, 8, 8).seed(seed);
        switch (a) {
            case CNN -> b.conv2d(16, 3, Activation.RELU).conv2d(32, 3, Activation.RELU).layer(GlobalPool.avg());
            case DEFORMABLE_CNN -> b.conv2d(16, 3, Activation.RELU).add(DeformableConv2d.spec(32, 3))
                    .activation(Activation.RELU).layer(GlobalPool.avg());
            case CONVNEXT -> b.conv2d(16, 3, Activation.IDENTITY).add(ConvNeXtBlock.spec(3))
                    .add(ConvNeXtBlock.spec(3)).layer(GlobalPool.avg());
            case VIT -> b.add(VisionTransformer.of(2, 32, 2, 4).block(TunableTransformerBlock.of(4).ffnFactor(2)));
            case VIT_ROPE_2D -> b.add(VisionTransformer.of(2, 32, 2, 4).rope2d(true).classToken(false)
                    .block(TunableTransformerBlock.of(4).ffnFactor(2).qkNorm(true)));
            case MLP_MIXER -> b.add(PatchEmbedding.spec(2, 32, false)).add(MlpMixerBlock.spec(2f, 2f))
                    .add(MlpMixerBlock.spec(2f, 2f)).layer(new SequencePool(SequencePool.Kind.MEAN));
        }
        return b.dense(ShapeImages.CLASSES, Activation.IDENTITY).build();
    }

    /**
     * Trains an architecture and returns its test accuracy.
     *
     * @param a the architecture
     * @param steps training steps
     * @param batch images per step
     * @param seed the seed of weights and data
     * @return the accuracy on 512 test images
     */
    public static float run(Architecture a, int steps, int batch, long seed) {
        Model model = build(a, seed);
        Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
        SplittableRandom rnd = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor[] data = IMAGES.generate(batch, rnd.nextLong());
            Functional.crossEntropy(model.forward(Var.of(data[0]), true), data[1]).backward();
            opt.step(model.parameters());
        }
        Tensor[] test = IMAGES.generate(512, seed + 99);
        return model.forward(Var.of(test[0])).value().argmax(1).equalTo(test[1]).mean().item();
    }
}
