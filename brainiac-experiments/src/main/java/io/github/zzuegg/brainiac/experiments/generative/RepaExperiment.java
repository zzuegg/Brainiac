package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.RepresentationAlignment;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.VectorConditionedDenoiser;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * REPA ({@link RepresentationAlignment}) on the {@link TinyDdpm} setting: 8x8
 * {@link ToyImages}, an epsilon-predicting MLP denoiser, deterministic DDIM sampling. A
 * small encoder is first trained self-supervised (SimCLR's NT-Xent from
 * {@link SelfSupervisedLosses}, views made by one-pixel shifts and pixel flips) and frozen;
 * then the same denoiser is trained twice from the same seed, once on the plain diffusion
 * loss and once with the alignment loss on its first hidden state, and sample quality
 * (mean squared distance per pixel to the nearest real image) is recorded along the way.
 */
@SuppressWarnings("exports")
public final class RepaExperiment {
    private static final int STEPS = 100;

    private RepaExperiment() {
    }

    private static float[] augment(float[] image, SplittableRandom r) {
        int dx = r.nextInt(3) - 1;
        int dy = r.nextInt(3) - 1;
        float[] out = new float[ToyImages.PIXELS];
        java.util.Arrays.fill(out, -1);
        for (int y = 0; y < ToyImages.SIDE; y++) {
            for (int x = 0; x < ToyImages.SIDE; x++) {
                int sx = x - dx;
                int sy = y - dy;
                if (sx >= 0 && sy >= 0 && sx < ToyImages.SIDE && sy < ToyImages.SIDE) {
                    out[y * ToyImages.SIDE + x] = image[sy * ToyImages.SIDE + sx];
                }
            }
        }
        for (int i = 0; i < out.length; i++) {
            if (r.nextDouble() < 0.05) {
                out[i] = -out[i];
            }
        }
        return out;
    }

    /**
     * Trains the frozen encoder.
     *
     * @param steps contrastive training steps
     * @param seed the seed
     * @return the encoder, 64 pixels to 32 features
     */
    public static Model encoder(int steps, long seed) {
        Model enc = Model.sequential().input(ToyImages.PIXELS).dense(128, Activation.TANH).dense(32).seed(seed)
                .build();
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            float[][] a = new float[64][];
            float[][] b = new float[64][];
            for (int i = 0; i < a.length; i++) {
                float[] img = ToyImages.sample(r.nextInt(ToyImages.CLASSES), r);
                a[i] = augment(img, r);
                b[i] = augment(img, r);
            }
            Var loss = SelfSupervisedLosses.ntXent(enc.forward(Var.of(Nets.matrix(a)), true),
                    enc.forward(Var.of(Nets.matrix(b)), true), 0.2f);
            loss.backward();
            opt.step(enc.parameters());
        }
        return enc;
    }

    /**
     * Trains a denoiser with or without alignment and scores it at checkpoints.
     *
     * @param encoder the frozen encoder, or {@code null} for plain training
     * @param checkpoints training steps (of 64 images) after which to score, ascending
     * @param lambda the alignment weight
     * @param seed the seed (the same for both runs)
     * @return the distance per pixel to the nearest real image at each checkpoint
     */
    public static double[] run(Model encoder, int[] checkpoints, float lambda, long seed) {
        VectorConditionedDenoiser net = VectorConditionedDenoiser.of(ToyImages.PIXELS, 0, 128, 3, seed)
                .timeScale(1f / STEPS);
        RepresentationAlignment repa = RepresentationAlignment.of(128, 32, 64, seed + 1);
        Diffusion d = new Diffusion(NoiseSchedule.cosine(STEPS), Diffusion.Prediction.NOISE).clipSample(1f);
        List<Var> params = new ArrayList<>(net.parameters());
        if (encoder != null) {
            params.addAll(repa.parameters());
        }
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed + 2);
        double[] out = new double[checkpoints.length];
        int done = 0;
        for (int c = 0; c < checkpoints.length; c++) {
            for (; done < checkpoints[c]; done++) {
                float[][] x = new float[64][];
                for (int b = 0; b < x.length; b++) {
                    x[b] = ToyImages.sample(r.nextInt(ToyImages.CLASSES), r);
                }
                Tensor x0 = Nets.matrix(x);
                Var loss = encoder == null ? d.loss(net.unconditional(), x0, r)
                        : repa.diffusionLoss(d, net, 1, x0, t -> encoder.forward(Var.of(t), false).value(), lambda,
                                r)[0];
                loss.backward();
                Nets.clip(params, 1f);
                opt.step(params);
            }
            out[c] = score(d, net, seed);
        }
        return out;
    }

    private static double score(Diffusion d, VectorConditionedDenoiser net, long seed) {
        Tensor noise = Nets.gaussian(128, ToyImages.PIXELS, new SplittableRandom(seed + 99));
        float[][] samples = Nets.rows(d.sample(net.unconditional(), noise, 10, 0, new SplittableRandom(seed + 3)));
        double dist = 0;
        for (float[] s : samples) {
            dist += ToyImages.nearest(s)[1] / ToyImages.PIXELS;
        }
        return dist / samples.length;
    }
}
