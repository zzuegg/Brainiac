package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * SimVP (Gao et al., 2022): video prediction with nothing but convolutions. A shared encoder
 * maps every context frame to a stride-2 feature map; the maps are stacked along channels
 * ({@code [batch, T * C, h, w]}) so a translator of residual 3x3 convolutions mixes time and
 * space at once and outputs the features of all future frames together; a shared decoder
 * (upsampling and convolutions) turns each into a frame. No recurrence, so all future frames
 * come out in one pass, which trains much faster than a ConvLSTM.
 *
 * <p>As a {@link Layer} it maps {@code [batch, contextFrames, h, w]} (frames as channels) to
 * {@code [batch, futureFrames, h, w]}.
 *
 * <p>Paper: Gao et al., "SimVP: Simpler yet Better Video Prediction" (2022),
 * <a href="https://arxiv.org/abs/2206.05099">arXiv:2206.05099</a>.
 */
public final class SimVp implements Layer {

    private final int context;
    private final int future;
    private final int channels;
    private final Var enc1;
    private final Var enc1b;
    private final Var enc2;
    private final Var enc2b;
    private final List<Var[]> translator = new ArrayList<>();
    private final Var mix;
    private final Var mixb;
    private final Var dec1;
    private final Var dec1b;
    private final Var dec2;
    private final Var dec2b;

    /**
     * Creates the model.
     *
     * @param context the context frames
     * @param future the predicted frames
     * @param channels the per-frame feature channels
     * @param blocks the residual translator blocks
     * @param random the source of initial weights
     */
    public SimVp(int context, int future, int channels, int blocks, RandomGenerator random) {
        this.context = context;
        this.future = future;
        this.channels = channels;
        this.enc1 = VisionOps.convWeight(random, channels, 1, 3);
        this.enc1b = VisionOps.zeros(channels);
        this.enc2 = VisionOps.convWeight(random, channels, channels, 3);
        this.enc2b = VisionOps.zeros(channels);
        int width = context * channels;
        for (int b = 0; b < blocks; b++) {
            translator.add(new Var[] {VisionOps.convWeight(random, width, width, 3), VisionOps.zeros(width)});
        }
        this.mix = VisionOps.convWeight(random, future * channels, width, 1);
        this.mixb = VisionOps.zeros(future * channels);
        this.dec1 = VisionOps.convWeight(random, channels, channels, 3);
        this.dec1b = VisionOps.zeros(channels);
        this.dec2 = VisionOps.convWeight(random, 1, 2 * channels, 3);
        this.dec2b = VisionOps.zeros(1);
    }

    /**
     * Describes the model for model builders.
     *
     * @param future the predicted frames
     * @param channels the per-frame channels
     * @param blocks the translator blocks
     * @return the spec, from {@code [context, h, w]} to {@code [future, h, w]}
     */
    public static LayerSpec spec(int future, int channels, int blocks) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(future, input.dim(1), input.dim(2));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SimVp(input.dim(0), future, channels, blocks, random);
            }
        };
    }

    @Override
    public Var forward(Var frames) {
        int b = frames.shape().dim(0);
        int h = frames.shape().dim(2);
        int w = frames.shape().dim(3);
        Var x = frames.reshape(b * context, 1, h, w);
        Var skip = VisionOps.conv(x, enc1, enc1b).relu();
        Var z = VisionOps.conv(skip, enc2, enc2b, 2, 1).relu();
        int h2 = z.shape().dim(2);
        int w2 = z.shape().dim(3);
        Var t = z.reshape(b, context * channels, h2, w2);
        for (Var[] blk : translator) {
            t = t.add(VisionOps.conv(t, blk[0], blk[1]).relu());
        }
        Var y = VisionOps.conv(t, mix, mixb).reshape(b * future, channels, h2, w2);
        y = VisionOps.conv(VisionOps.upsample(y, h, w), dec1, dec1b).relu();
        // the last context frame's encoder features as a skip for every future frame
        Var lastSkip = skip.reshape(b, context, channels, h, w).slice(1, context - 1, context)
                .expand(Shape.of(b, future, channels, h, w)).reshape(b * future, channels, h, w);
        return VisionOps.conv(Var.concat(1, y, lastSkip), dec2, dec2b).reshape(b, future, h, w);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(future, input.dim(1), input.dim(2));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(enc1, enc1b, enc2, enc2b));
        for (Var[] blk : translator) {
            p.add(blk[0]);
            p.add(blk[1]);
        }
        p.addAll(List.of(mix, mixb, dec1, dec1b, dec2, dec2b));
        return p;
    }

    @Override
    public String toString() {
        return "SimVp(" + context + " -> " + future + " frames)";
    }
}
