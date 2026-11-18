package io.github.zzuegg.brainiac.experimental.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Self Forcing (Huang et al., NeurIPS 2025, <a href="https://arxiv.org/abs/2506.08009">arXiv:2506.08009</a>):
 * training an autoregressive video diffusion model on its own rollouts. Teacher forcing (clean
 * history) and diffusion forcing (noisy history) never show the model its own generated history
 * during training, so errors compound at inference. Here every training step
 *
 * <ol>
 *   <li>rolls the few-step generator out frame by frame, exactly as at inference: each new
 *       frame is denoised in {@code k} steps (predict {@code x_0}, re-noise to the next level)
 *       from the history of frames <em>it generated itself</em>, held in a rolling history
 *       cache;
 *   <li>bounds memory by back-propagating only through one denoising step per frame: the
 *       number of steps {@code s <= k} is drawn at random, the first {@code s - 1} run without
 *       a graph and the frame is the {@code x_0} prediction of step {@code s}; the cached
 *       history is detached, so no gradient flows into earlier frames;
 *   <li>scores the whole generated clip with a holistic distribution-matching (DMD) loss:
 *       with the clip diffused to a random level {@code tau}, the update direction is the
 *       difference between the clean estimates of a frozen bidirectional video model trained
 *       on real clips (the real score) and of a second video model trained online on the
 *       generator's clips (the fake score), normalised per clip by the real estimate's error,
 *       as in {@code DistributionMatchingDistiller}. The fake model then takes its own flow
 *       matching step on the detached clip.
 * </ol>
 *
 * <p>The video-level scores are {@link SpaceTimeDenoiser}s conditioned on the given first frame,
 * so the loss compares whole clips, not single frames. {@code DistributionMatchingDistiller}
 * itself scores unconditional samples through its {@code Denoiser} interface and has no way
 * to condition on a prefix or to take a per-frame-level clip, which is why the DMD step is
 * written out here.
 */
public final class SelfForcingTrainer {

    private final FrameDenoiser generator;
    private final SpaceTimeDenoiser real;
    private final SpaceTimeDenoiser fake;
    private final int frames;
    private final int steps;
    private final Optimizer generatorOptimizer;
    private final Optimizer fakeOptimizer;
    private int fakeUpdates = 1;

    private SelfForcingTrainer(FrameDenoiser generator, SpaceTimeDenoiser real, SpaceTimeDenoiser fake, int frames,
            int steps, float generatorRate, float fakeRate) {
        this.generator = generator;
        this.real = real;
        this.fake = fake;
        this.frames = frames;
        this.steps = steps;
        this.generatorOptimizer = Optimizer.adam(generatorRate);
        this.fakeOptimizer = Optimizer.adam(fakeRate);
    }

    /**
     * Creates a trainer.
     *
     * @param generator the autoregressive model, usually pretrained by teacher or diffusion forcing
     * @param real the frozen bidirectional video model of real clips (first-frame conditioned)
     * @param fake the online video model of generated clips, same architecture
     * @param frames the training clip length
     * @param steps the generator's denoising steps per frame {@code k}
     * @param generatorRate the generator's learning rate
     * @param fakeRate the fake score's learning rate
     * @return the trainer
     */
    public static SelfForcingTrainer of(FrameDenoiser generator, SpaceTimeDenoiser real, SpaceTimeDenoiser fake,
            int frames, int steps, float generatorRate, float fakeRate) {
        return new SelfForcingTrainer(generator, real, fake, frames, steps, generatorRate, fakeRate);
    }

    /**
     * @param updates fake-score updates per generator update (DMD2's two time-scale rule)
     * @return this trainer
     */
    public SelfForcingTrainer fakeUpdates(int updates) {
        this.fakeUpdates = updates;
        return this;
    }

    /**
     * One few-step denoising of the next frame with a graph through the last executed step only.
     *
     * @param history the detached history {@code [b, W, side, side]}
     * @param available existing history frames
     * @param random the random source
     * @return the frame's {@code x_0} estimate, with a graph to the generator's parameters
     */
    Var generateFrame(Tensor history, int available, SplittableRandom random) {
        int b = history.shape().dim(0);
        int side = generator.side();
        Shape s = Shape.of(b, 1, side, side);
        int stop = 1 + random.nextInt(steps);
        float[] hl = generator.historyLevels(b, available, 0);
        float[] x = new float[b * side * side];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) random.nextGaussian();
        }
        for (int k = 0; k < stop - 1; k++) {
            float level = 1 - (float) k / steps;
            float nextLevel = 1 - (float) (k + 1) / steps;
            Tensor xt = Tensor.of(s, x);
            float[] v = Var.noGrad(() -> generator.velocity(history, hl, xt, level, random)).value().toArray();
            for (int i = 0; i < x.length; i++) {
                float x0 = x[i] - level * v[i];
                x[i] = (1 - nextLevel) * x0 + nextLevel * (float) random.nextGaussian();
            }
        }
        float level = 1 - (float) (stop - 1) / steps;
        Tensor xt = Tensor.of(s, x);
        Var v = generator.velocity(history, hl, xt, level, random);
        return v.mul(-level).add(xt);
    }

    /**
     * Generates clips with the few-step rule (no graph), for evaluation and long rollouts.
     *
     * @param first the given first frames {@code [b, 1, side, side]}
     * @param total the clip length
     * @param random the noise source
     * @return {@code [b, total, side, side]}
     */
    public Tensor generate(Tensor first, int total, SplittableRandom random) {
        Tensor video = first;
        int b = first.shape().dim(0);
        int stepsFull = steps;
        while (video.shape().dim(1) < total) {
            int have = video.shape().dim(1);
            int available = Math.min(generator.window(), have);
            Tensor history = generator.history(video, available, b);
            Tensor frame = Var.noGrad(() -> fullSteps(history, available, stepsFull, random)).value();
            video = Tensor.concat(1, video, frame);
        }
        return video;
    }

    private Var fullSteps(Tensor history, int available, int k, SplittableRandom random) {
        int b = history.shape().dim(0);
        int side = generator.side();
        Shape s = Shape.of(b, 1, side, side);
        float[] hl = generator.historyLevels(b, available, 0);
        float[] x = new float[b * side * side];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) random.nextGaussian();
        }
        Var out = null;
        for (int j = 0; j < k; j++) {
            float level = 1 - (float) j / k;
            float nextLevel = 1 - (float) (j + 1) / k;
            Tensor xt = Tensor.of(s, x);
            float[] v = generator.velocity(history, hl, xt, level, random).value().toArray();
            float[] x0 = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                x0[i] = x[i] - level * v[i];
                x[i] = (1 - nextLevel) * x0[i] + nextLevel * (float) random.nextGaussian();
            }
            out = Var.of(Tensor.of(s, x0));
        }
        return out;
    }

    /**
     * One Self Forcing step.
     *
     * @param first real first frames {@code [b, 1, side, side]} to condition the rollout on
     * @param random the random source
     * @return {@code {generator loss, fake-score loss}}
     */
    public float[] step(Tensor first, SplittableRandom random) {
        int b = first.shape().dim(0);
        int side = generator.side();
        List<Var> clip = new ArrayList<>();
        clip.add(Var.of(first));
        Tensor video = first;
        for (int f = 1; f < frames; f++) {
            int available = Math.min(generator.window(), f);
            Tensor history = generator.history(video, available, b);
            Var frame = generateFrame(history, available, random);
            clip.add(frame);
            video = Tensor.concat(1, video, frame.value());
        }
        Var generated = Var.concat(1, clip.toArray(new Var[0]));
        // holistic DMD on the whole clip
        float tau = 0.02f + 0.96f * random.nextFloat();
        Shape s = Shape.of(b, frames, side, side);
        float[] x = video.toArray();
        float[] xt = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            xt[i] = (1 - tau) * x[i] + tau * (float) random.nextGaussian();
        }
        int pix = side * side;
        float[] lv = new float[b * frames];
        for (int i = 0; i < b; i++) {
            for (int f = 1; f < frames; f++) {
                lv[i * frames + f] = tau;
            }
            System.arraycopy(first.toArray(), i * pix, xt, i * frames * pix, pix);
        }
        Tensor noisy = Tensor.of(s, xt);
        Tensor levels = Tensor.of(Shape.of(b, frames), lv);
        float[] vReal = Var.noGrad(() -> real.apply(Var.of(noisy), levels)).value().toArray();
        float[] vFake = Var.noGrad(() -> fake.apply(Var.of(noisy), levels)).value().toArray();
        float[] target = new float[x.length];
        float[] mask = new float[x.length];
        for (int i = 0; i < b; i++) {
            double norm = 0;
            for (int f = 1; f < frames; f++) {
                for (int p = 0; p < pix; p++) {
                    int idx = (i * frames + f) * pix + p;
                    norm += Math.abs(x[idx] - (xt[idx] - tau * vReal[idx]));
                }
            }
            norm = Math.max(1e-6, norm / ((frames - 1) * pix));
            for (int f = 1; f < frames; f++) {
                for (int p = 0; p < pix; p++) {
                    int idx = (i * frames + f) * pix + p;
                    double x0Real = xt[idx] - tau * vReal[idx];
                    double x0Fake = xt[idx] - tau * vFake[idx];
                    target[idx] = (float) (x[idx] - (x0Fake - x0Real) / norm);
                    mask[idx] = 1;
                }
            }
        }
        Var loss = generated.sub(Tensor.of(s, target)).square().mul(Tensor.of(s, mask)).sum()
                .mul(0.5f / (b * (frames - 1) * pix));
        float gl = loss.value().item();
        loss.backward();
        generatorOptimizer.step(generator.parameters());
        float fl = 0;
        int[] context = new int[b];
        java.util.Arrays.fill(context, 1);
        for (int u = 0; u < fakeUpdates; u++) {
            float level = random.nextFloat();
            float[] fl2 = new float[b * frames];
            for (int i = 0; i < b; i++) {
                for (int f = 1; f < frames; f++) {
                    fl2[i * frames + f] = level;
                }
            }
            Var l = fake.flowLoss(video, fl2, context, random);
            fl = l.value().item();
            l.backward();
            fakeOptimizer.step(fake.parameters());
        }
        return new float[] {gl, fl};
    }
}
