package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ControlBranch;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * ControlNet against full fine-tuning and condition concatenation (wave 6, item 4.10) on 8x8
 * {@link ToyImages}. A residual MLP denoiser ({@link ControlBranch.Backbone}, core
 * {@link Diffusion}, cosine schedule, 100 steps) is pretrained unconditionally; the control is
 * the coarse 4x4 occupancy map of the target image (where the shape is). With the same number
 * of control-training steps:
 * <ul>
 *   <li><b>ControlNet</b>: {@link ControlBranch} on the frozen base;</li>
 *   <li><b>fine-tune</b>: the base with a zero-initialised condition input, every weight trained;</li>
 *   <li><b>concat</b>: a fresh backbone with the condition concatenated to its input, from scratch.</li>
 * </ul>
 * Scores: control adherence (IoU of the generated image's occupancy with the control), the
 * controlled samples' quality ({@link GenerativeScores#images}), and base-quality retention: the
 * unconditional samples (control branch off, or a zero condition) after control training.
 */
@SuppressWarnings("exports")
public final class ControlExperiment {
    private static final int T = 100;
    private static final int CELLS = 16;
    private static final Diffusion DIFFUSION = new Diffusion(NoiseSchedule.cosine(T), Diffusion.Prediction.NOISE)
            .clipSample(1f);

    private ControlExperiment() {
    }

    /** The 4x4 occupancy of an image: 1 where any pixel of the 2x2 cell is on. */
    static float[] occupancy(float[] image) {
        float[] c = new float[CELLS];
        for (int p = 0; p < ToyImages.PIXELS; p++) {
            if (image[p] > 0) {
                c[(p / ToyImages.SIDE / 2) * 4 + (p % ToyImages.SIDE) / 2] = 1;
            }
        }
        return c;
    }

    private static Tensor controls(float[][] images) {
        float[][] c = new float[images.length][];
        for (int i = 0; i < c.length; i++) {
            c[i] = occupancy(images[i]);
        }
        return Nets.matrix(c);
    }

    private static void train(Denoiser.Conditional model, List<Var> params, int steps, float lr, boolean condition,
            SplittableRandom r) {
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            float[][] x = PointClouds.images(64, r);
            Tensor c = condition ? controls(x) : null;
            DIFFUSION.loss(model.given(c), Nets.matrix(x), r).backward();
            Nets.clip(params, 5f);
            opt.step(params);
        }
    }

    private static double iou(float[][] samples, Tensor control) {
        float[] c = control.toArray();
        double total = 0;
        for (int i = 0; i < samples.length; i++) {
            float[] o = occupancy(samples[i]);
            int inter = 0;
            int union = 0;
            for (int k = 0; k < CELLS; k++) {
                boolean a = o[k] > 0;
                boolean b = c[i * CELLS + k] > 0;
                inter += a && b ? 1 : 0;
                union += a || b ? 1 : 0;
            }
            total += union == 0 ? 1 : (double) inter / union;
        }
        return total / samples.length;
    }

    private static float[][] sample(Denoiser model, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return Nets.rows(DIFFUSION.sample(model, Nets.gaussian(128, ToyImages.PIXELS, r), 20, 0, r));
    }

    /**
     * Pretrains the base, trains the three conditioned models and scores them.
     *
     * @param baseSteps unconditional pretraining steps of 64 images
     * @param controlSteps control-training steps of 64 images for every method
     * @param seed the seed
     * @return per method ({@code base}, {@code ControlNet}, {@code fine-tune}, {@code concat}):
     *     {@code "<method> IoU"}, {@code "<method> distance"}, {@code "<method> real"}, and for
     *     the base and the two methods that reuse it, {@code "<method> unconditional distance"}
     */
    public static Map<String, Double> run(int baseSteps, int controlSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        ControlBranch.Backbone base = ControlBranch.Backbone.of(ToyImages.PIXELS, 128, 3, 0, 1f / T, seed);
        train(base, base.parameters(), baseSteps, 2e-3f, false, r);
        float[][] baseUncond = sample(base, seed + 5);

        ControlBranch control = ControlBranch.of(base, CELLS);
        train(control, control.parameters(), controlSteps, 2e-3f, true, r);
        ControlBranch.Backbone fineTune = base.withCondition(CELLS);
        train(fineTune, fineTune.parameters(), controlSteps, 2e-3f, true, r);
        ControlBranch.Backbone concat = ControlBranch.Backbone.of(ToyImages.PIXELS, 128, 3, CELLS, 1f / T, seed + 1);
        train(concat, concat.parameters(), controlSteps, 2e-3f, true, r);

        Tensor targets = controls(PointClouds.images(128, new SplittableRandom(seed + 9)));
        Tensor none = Tensor.zeros(targets.shape());
        Map<String, Double> out = new LinkedHashMap<>();
        put(out, "base", base, targets);
        out.put("base unconditional distance", GenerativeScores.images(baseUncond).distance());
        put(out, "ControlNet", control, targets);
        out.put("ControlNet unconditional distance", GenerativeScores.images(sample(control.base(), seed + 5))
                .distance());
        put(out, "fine-tune", fineTune, targets);
        out.put("fine-tune unconditional distance", GenerativeScores.images(sample(fineTune.given(none), seed + 5))
                .distance());
        put(out, "concat", concat, targets);
        return out;
    }

    private static void put(Map<String, Double> out, String name, Denoiser.Conditional model, Tensor targets) {
        float[][] s = sample(model.given(targets), 77);
        GenerativeScores.ImageScore score = GenerativeScores.images(s);
        out.put(name + " IoU", iou(s, targets));
        out.put(name + " distance", score.distance());
        out.put(name + " real", score.realShare());
    }
}
