package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.RealNvpFlow;
import io.github.zzuegg.brainiac.experimental.nn.TransformerFlow;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A {@link TransformerFlow} (TarFlow, <a href="https://arxiv.org/abs/2412.06329">arXiv:2412.06329</a>)
 * on {@link ToyImages} cut into sixteen 2x2 patches, against a coupling flow
 * ({@link RealNvpFlow}) on the same noise-augmented data, by held-out negative
 * log-likelihood in bits per dimension (density in 8-bit pixel units), and against
 * {@link TinyDdpm} by sample quality: the squared distance of samples to the nearest real
 * image. TarFlow samples are scored before and after the score-based denoising step.
 */
public final class TarFlowExperiment {
    private static final int SIDE = ToyImages.SIDE;

    private TarFlowExperiment() {
    }

    /** Pixel order: 2x2 patches in raster order, pixels within a patch in raster order. */
    private static float[] toPatches(float[] image) {
        float[] out = new float[image.length];
        int k = 0;
        for (int pr = 0; pr < SIDE; pr += 2) {
            for (int pc = 0; pc < SIDE; pc += 2) {
                for (int r = 0; r < 2; r++) {
                    for (int c = 0; c < 2; c++) {
                        out[k++] = image[(pr + r) * SIDE + pc + c];
                    }
                }
            }
        }
        return out;
    }

    private static float[] fromPatches(float[] patches) {
        float[] out = new float[patches.length];
        int k = 0;
        for (int pr = 0; pr < SIDE; pr += 2) {
            for (int pc = 0; pc < SIDE; pc += 2) {
                for (int r = 0; r < 2; r++) {
                    for (int c = 0; c < 2; c++) {
                        out[(pr + r) * SIDE + pc + c] = patches[k++];
                    }
                }
            }
        }
        return out;
    }

    private static float[][] noisyBatch(SplittableRandom r, int n, float sigma) {
        float[][] x = new float[n][];
        for (int b = 0; b < n; b++) {
            x[b] = toPatches(ToyImages.sample(r.nextInt(ToyImages.CLASSES), r));
            for (int i = 0; i < x[b].length; i++) {
                x[b][i] += (float) (sigma * r.nextGaussian());
            }
        }
        return x;
    }

    /**
     * Trains the three models and measures them.
     *
     * @param iterations training steps of 32 images for each model
     * @param sigma the noise added to training data of both flows
     * @param seed the seed
     * @return metrics: {@code "<model> bpd"} and {@code "<model> distance"}
     */
    public static Map<String, Double> run(int iterations, float sigma, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        TransformerFlow tar = new TransformerFlow(16, 4, 2, 32, 1, 2, seed);
        RealNvpFlow nvp = new RealNvpFlow(ToyImages.PIXELS, 8, 64, new SplittableRandom(seed));
        TinyDdpm ddpm = TinyDdpm.of(MlpDenoiser.of(ToyImages.PIXELS, 0, 128, 2, seed).timeScale(1f / 50), 50, true,
                3e-3f, 0, seed);
        Optimizer tarOpt = Optimizer.adam(2e-3f);
        Optimizer nvpOpt = Optimizer.adam(1e-3f);
        int batch = 32;
        int[] none = new int[batch];
        Arrays.fill(none, -1);
        for (int i = 0; i < iterations; i++) {
            Tensor x = Nets.matrix(noisyBatch(r, batch, sigma));
            Var tl = tar.logProb(Var.of(x)).mean().neg();
            tl.backward();
            Nets.clip(tar.parameters(), 1f);
            tarOpt.step(tar.parameters());
            Var nl = nvp.logProb(Var.of(x)).mean().neg();
            nl.backward();
            Nets.clip(nvp.parameters(), 1f);
            nvpOpt.step(nvp.parameters());
            float[][] clean = new float[batch][];
            for (int b = 0; b < batch; b++) {
                clean[b] = ToyImages.sample(r.nextInt(ToyImages.CLASSES), r);
            }
            ddpm.trainStep(clean, none);
        }
        Tensor held = Nets.matrix(noisyBatch(new SplittableRandom(seed + 100), 256, sigma));
        double toBits = 1 / (ToyImages.PIXELS * Math.log(2));
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("TarFlow bpd", -tar.logProb(Var.of(held)).mean().value().item() * toBits + 7);
        out.put("RealNVP bpd", -nvp.logProb(Var.of(held)).mean().value().item() * toBits + 7);
        Tensor raw = tar.sample(64, new SplittableRandom(seed + 7));
        out.put("TarFlow raw distance", distance(raw, true));
        out.put("TarFlow distance", distance(tar.denoise(raw, sigma), true));
        out.put("RealNVP distance", distance(nvp.sample(64, new SplittableRandom(seed + 7)), true));
        int[] labels = new int[64];
        Arrays.fill(labels, -1);
        out.put("DDPM distance", distance(Nets.matrix(ddpm.sampleDdim(labels, 20, 1f, seed + 7, null)), false));
        return out;
    }

    private static double distance(Tensor samples, boolean patched) {
        float[][] rows = Nets.rows(samples);
        double sum = 0;
        for (float[] s : rows) {
            sum += ToyImages.nearest(patched ? fromPatches(s) : s)[1];
        }
        return sum / rows.length;
    }
}
