package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.ConvNeXtBlock;
import io.github.zzuegg.brainiac.experimental.nn.DeformableConv2d;
import io.github.zzuegg.brainiac.experimental.nn.MlpMixerBlock;
import io.github.zzuegg.brainiac.experimental.nn.PatchEmbedding;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock;
import io.github.zzuegg.brainiac.experimental.nn.VisionTransformer;
import io.github.zzuegg.brainiac.experimental.vision.FlexiblePatchEmbedding;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.GlobalPool;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.SequencePool;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
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
 *
 * <p>{@link #flexiViT} (wave 10, 10.6.9) trains one ViT whose patch embedding is a
 * {@link FlexiblePatchEmbedding} with a random patch size per step and evaluates it at patch
 * sizes 2, 4 and 8, against separate ViTs per patch size and a patch-4 ViT evaluated at the
 * other sizes with naive (bilinear) and pseudo-inverse weight resizing.
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

    // ---------------------------------------------------------------- 10.6.9 FlexiViT

    /** A small ViT classifier on a {@link FlexiblePatchEmbedding}. */
    static final class FlexibleVit {
        final FlexiblePatchEmbedding embed;
        final List<Layer> blocks = new ArrayList<>();
        final Var head;
        final Var headBias;

        FlexibleVit(int side, FlexiblePatchEmbedding.Resize resize, SplittableRandom r) {
            embed = new FlexiblePatchEmbedding(1, 32, 4, side / 4, new int[] {2, 4, 8}, resize, r);
            for (int i = 0; i < 2; i++) {
                blocks.add(TransformerBlock.of(4).ffnSize(64).create(Shape.of(16, 32), r));
            }
            head = Var.param(Tensor.normal(Shape.of(32, ShapeImages.CLASSES), 0, 0.18f, r));
            headBias = Var.param(Tensor.zeros(Shape.of(ShapeImages.CLASSES)));
        }

        Var forward(Var x, boolean training) {
            Var h = embed.forward(x);
            for (Layer b : blocks) {
                h = b.forward(h, training);
            }
            return h.mean(1, false).matmul(head).add(headBias);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(embed.parameters());
            for (Layer b : blocks) {
                p.addAll(b.parameters());
            }
            p.add(head);
            p.add(headBias);
            return p;
        }

        long macs(int side, int patch) {
            long n = (long) (side / patch) * (side / patch);
            return embed.macs(side, side) + 2 * (n * (4L * 32 * 32 + 2L * 32 * 64) + 2 * n * n * 32);
        }
    }

    /**
     * One FlexiViT comparison row.
     *
     * @param method the training scheme
     * @param patch the evaluation patch size
     * @param accuracy test accuracy
     * @param macs multiply-adds per image
     */
    public record FlexiRow(String method, int patch, double accuracy, long macs) {
        @Override
        public String toString() {
            return String.format("%s@p%d %.3f (%.2fM MACs)", method, patch, accuracy, macs / 1e6);
        }
    }

    private static void trainVit(FlexibleVit vit, ShapeImages images, int steps, int fixedPatch, long seed) {
        Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
        SplittableRandom rnd = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            if (fixedPatch > 0) {
                vit.embed.patch(fixedPatch);
            } else {
                vit.embed.samplePatch(rnd);
            }
            Tensor[] data = images.generate(32, rnd.nextLong());
            Functional.crossEntropy(vit.forward(Var.of(data[0]), true), data[1]).backward();
            opt.step(vit.parameters());
        }
    }

    private static double accuracy(FlexibleVit vit, Tensor[] test, int patch) {
        vit.embed.patch(patch);
        return Var.noGrad(() -> vit.forward(Var.of(test[0]), false)).value().argmax(1).equalTo(test[1]).mean().item();
    }

    /**
     * FlexiViT against fixed-patch ViTs on {@code side x side} {@link ShapeImages}.
     *
     * @param side the image side (divisible by 8)
     * @param steps the total training budget of each scheme (the separate ViTs share it three
     *     ways)
     * @param fullSeparate whether to also train separate ViTs with the full budget each
     * @param seed the seed
     * @return rows per scheme and patch size
     */
    public static List<FlexiRow> flexiViT(int side, int steps, boolean fullSeparate, long seed) {
        ShapeImages images = new ShapeImages(side, 0.1f);
        Tensor[] test = images.generate(256, seed + 99);
        int[] patches = {2, 4, 8};
        List<FlexiRow> rows = new ArrayList<>();
        FlexibleVit flexi = new FlexibleVit(side, FlexiblePatchEmbedding.Resize.PSEUDO_INVERSE, new SplittableRandom(seed));
        trainVit(flexi, images, steps, 0, seed);
        for (int p : patches) {
            rows.add(new FlexiRow("FlexiViT", p, accuracy(flexi, test, p), flexi.macs(side, p)));
        }
        for (int p : patches) {
            FlexibleVit v = new FlexibleVit(side, FlexiblePatchEmbedding.Resize.PSEUDO_INVERSE, new SplittableRandom(seed));
            trainVit(v, images, steps / 3, p, seed);
            rows.add(new FlexiRow("separate (1/3 budget)", p, accuracy(v, test, p), v.macs(side, p)));
            if (fullSeparate) {
                FlexibleVit full = new FlexibleVit(side, FlexiblePatchEmbedding.Resize.PSEUDO_INVERSE,
                        new SplittableRandom(seed));
                trainVit(full, images, steps, p, seed);
                rows.add(new FlexiRow("separate (full budget)", p, accuracy(full, test, p), full.macs(side, p)));
            }
        }
        FlexibleVit naive = new FlexibleVit(side, FlexiblePatchEmbedding.Resize.BILINEAR, new SplittableRandom(seed));
        trainVit(naive, images, steps, 4, seed);
        FlexibleVit pi = new FlexibleVit(side, FlexiblePatchEmbedding.Resize.PSEUDO_INVERSE, new SplittableRandom(seed));
        trainVit(pi, images, steps, 4, seed);
        for (int p : patches) {
            rows.add(new FlexiRow("patch-4 ViT, naive resize", p, accuracy(naive, test, p), naive.macs(side, p)));
            rows.add(new FlexiRow("patch-4 ViT, PI resize", p, accuracy(pi, test, p), pi.macs(side, p)));
        }
        return rows;
    }

    /**
     * Prints the long-budget FlexiViT run (32x32 images).
     *
     * @param args unused
     */
    public static void main(String[] args) {
        System.out.println("FlexiViT 32x32: " + flexiViT(32, 3000, true, 1));
    }
}
