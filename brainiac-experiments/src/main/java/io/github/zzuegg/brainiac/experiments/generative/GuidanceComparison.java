package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.AdaptiveProjectedGuidance;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.Autoguidance;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.ConditionalDenoiser;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Guidance rules compared at matched strength on {@link ToyPoints} (a ring of 8 Gaussian
 * blobs, class = blob), with a class-conditional noise-predicting diffusion model whose
 * outputs are turned into clean-sample estimates (the space APG is defined in):
 * classifier-free guidance ({@link ConditionalDenoiser#guided}), adaptive projected
 * guidance ({@link AdaptiveProjectedGuidance}) and autoguidance by a small, briefly trained
 * model ({@link Autoguidance}). Samples are scored by improved precision and recall
 * (Kynkaanniemi et al., <a href="https://arxiv.org/abs/1904.06991">1904.06991</a>; k-NN
 * balls, k = 3) against real points of the requested classes: precision is the share of
 * samples inside the real data's balls, recall the share of real points inside the
 * samples' balls.
 */
@SuppressWarnings("exports")
public final class GuidanceComparison {
    private static final int STEPS = 100;
    private static final ToyPoints DATA = ToyPoints.ring(8, 2, 0.2);

    private GuidanceComparison() {
    }

    private static MlpDenoiser train(int width, int depth, int iterations, long seed) {
        MlpDenoiser net = MlpDenoiser.of(2, DATA.modes(), width, depth, seed).timeScale(1f / STEPS);
        Diffusion d = new Diffusion(NoiseSchedule.cosine(STEPS), Diffusion.Prediction.NOISE);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int i = 0; i < iterations; i++) {
            float[][] x = new float[128][];
            int[] labels = new int[128];
            for (int b = 0; b < x.length; b++) {
                labels[b] = r.nextInt(DATA.modes());
                x[b] = DATA.sample(labels[b], r);
            }
            Var loss = d.loss(net.bind(ConditionalDenoiser.dropLabels(labels, 0.2, r)), Nets.matrix(x), r);
            loss.backward();
            Nets.clip(net.parameters(), 1f);
            opt.step(net.parameters());
        }
        return net;
    }

    /**
     * Improved precision and recall of samples against real points.
     *
     * @param real real points
     * @param fake generated points
     * @param k the neighbour defining each ball's radius
     * @return {@code {precision, recall}}
     */
    public static double[] precisionRecall(float[][] real, float[][] fake, int k) {
        return new double[] {coverage(real, fake, k), coverage(fake, real, k)};
    }

    /** The share of {@code queries} inside some ball around {@code support} (radius: k-th neighbour in support). */
    private static double coverage(float[][] support, float[][] queries, int k) {
        double[] radius = new double[support.length];
        for (int i = 0; i < support.length; i++) {
            double[] d = new double[support.length];
            for (int j = 0; j < support.length; j++) {
                d[j] = Nets.distance2(support[i], support[j]);
            }
            Arrays.sort(d);
            radius[i] = d[Math.min(k, d.length - 1)];
        }
        int in = 0;
        for (float[] q : queries) {
            for (int i = 0; i < support.length; i++) {
                if (Nets.distance2(q, support[i]) <= radius[i]) {
                    in++;
                    break;
                }
            }
        }
        return (double) in / queries.length;
    }

    /**
     * Trains the models once and samples every rule at every weight.
     *
     * @param iterations training steps of the main model
     * @param weights guidance weights to try
     * @param seed the seed
     * @return per rule and weight, {@code {precision, recall}}; also "none" (w = 1)
     */
    public static Map<String, double[]> run(int iterations, float[] weights, long seed) {
        MlpDenoiser good = train(128, 2, iterations, seed);
        MlpDenoiser bad = train(64, 1, iterations / 3, seed + 10);
        Denoiser.Conditional goodX = clean(good);
        Denoiser.Conditional badX = clean(bad);
        int n = 400;
        int[] labels = new int[n];
        float[][] real = new float[n][];
        SplittableRandom r = new SplittableRandom(seed + 5);
        for (int i = 0; i < n; i++) {
            labels[i] = i % DATA.modes();
            real[i] = DATA.sample(labels[i], r);
        }
        int[] none = new int[n];
        Arrays.fill(none, -1);
        Tensor cond = ConditionalDenoiser.condition(labels);
        Tensor uncond = ConditionalDenoiser.condition(none);
        Diffusion d = new Diffusion(NoiseSchedule.cosine(STEPS), Diffusion.Prediction.SAMPLE);   // on clean estimates
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("none", precisionRecall(real, sample(d, goodX.given(cond), seed), 3));
        for (float w : weights) {
            out.put("CFG w=" + w, precisionRecall(real, sample(d, goodX.guided(cond, uncond, w), seed), 3));
            out.put("APG w=" + w, precisionRecall(real, sample(d,
                    AdaptiveProjectedGuidance.of(w, 0, 1f, -0.5f).guided(goodX, cond, uncond), seed), 3));
            out.put("autoguidance w=" + w, precisionRecall(real, sample(d,
                    Autoguidance.guide(goodX.given(cond), badX.given(cond), w), seed), 3));
        }
        return out;
    }

    /** A noise-predicting model seen as predicting the clean sample (the space APG works in). */
    private static Denoiser.Conditional clean(MlpDenoiser net) {
        NoiseSchedule schedule = NoiseSchedule.cosine(STEPS);
        return (noisy, time, condition) -> {
            float[] t = time.toArray();
            float[] a = new float[t.length];
            float[] b = new float[t.length];
            for (int i = 0; i < t.length; i++) {
                double ab = schedule.alphaBar(Math.round(t[i]));
                a[i] = (float) (1 / Math.sqrt(ab));
                b[i] = (float) Math.sqrt((1 - ab) / ab);
            }
            Var eps = net.apply(noisy, time, condition);
            return noisy.mul(Var.of(Nets.column(a))).sub(eps.mul(Var.of(Nets.column(b)))).clamp(-5, 5);
        };
    }

    private static float[][] sample(Diffusion d, Denoiser model, long seed) {
        Tensor noise = Nets.gaussian(400, 2, new SplittableRandom(seed + 99));
        return Nets.rows(d.sample(model, noise, 50, 0, new SplittableRandom(seed + 3)));
    }
}
