package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DenoisingThermodynamicModel;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Generative modelling on probabilistic bits: a {@link DenoisingThermodynamicModel} (DTM,
 * <a href="https://arxiv.org/abs/2510.23972">arXiv:2510.23972</a>) of chained p-bit denoising
 * stages against a single Boltzmann machine of the same size trained the same way (one
 * stage, given as many Gibbs sweeps in total for sampling), and against a continuous DDPM
 * (Ho et al., <a href="https://arxiv.org/abs/2006.11239">arXiv:2006.11239</a>) on the
 * {@code +-1} vectors, whose samples are thresholded at 0.
 *
 * <p>The data are the 30 4x4 "bars and stripes" patterns (MacKay, <em>Information Theory,
 * Inference, and Learning Algorithms</em>, 2003), a classic multimodal test for Boltzmann
 * machines. Scores: the share of samples that are valid patterns, how many of the 30 are
 * produced, and the maximum mean discrepancy to the data under the kernel
 * {@code exp(-hamming / 4)} (lower is better).
 */
public final class ThermodynamicDenoising {
    private static final int SIDE = 4;
    private static final int BITS = SIDE * SIDE;
    private static final int[][] PATTERNS = patterns();

    /**
     * Sample quality.
     *
     * @param method the method
     * @param validShare the share of valid patterns
     * @param distinct the number of distinct valid patterns produced, of 30
     * @param mmd the squared maximum mean discrepancy to the data
     */
    public record Score(String method, double validShare, int distinct, double mmd) {
    }

    private ThermodynamicDenoising() {
    }

    private static int[][] patterns() {
        Set<String> seen = new HashSet<>();
        int[][] out = new int[2 * (1 << SIDE) - 2][];
        int n = 0;
        for (int stripes = 0; stripes < 2; stripes++) {
            for (int mask = 0; mask < 1 << SIDE; mask++) {
                int[] p = new int[BITS];
                for (int y = 0; y < SIDE; y++) {
                    for (int x = 0; x < SIDE; x++) {
                        int line = stripes == 0 ? x : y;
                        p[y * SIDE + x] = (mask >> line & 1) == 1 ? 1 : -1;
                    }
                }
                if (seen.add(Arrays.toString(p))) {
                    out[n++] = p;
                }
            }
        }
        return out;
    }

    private static int[][] data(int n, SplittableRandom r) {
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = PATTERNS[r.nextInt(PATTERNS.length)];
        }
        return out;
    }

    /**
     * Trains a p-bit model with contrastive divergence and scores its samples.
     *
     * @param stages denoising stages (1 = single machine)
     * @param sweepBudget total Gibbs sweeps per sample, split evenly over the stages
     * @param steps training steps on batches of 32
     * @param seed the seed
     * @return the score
     */
    public static Score pbits(int stages, int sweepBudget, int steps, long seed) {
        DenoisingThermodynamicModel m = DenoisingThermodynamicModel.of(BITS, 16, stages, seed);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            m.train(data(32, r), 5, 0.05, r);
        }
        int[][] samples = new int[400][];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = m.sample(sweepBudget / stages, r);
        }
        return score(stages == 1 ? "single Boltzmann machine" : "DTM, " + stages + " stages", samples, seed);
    }

    /**
     * Trains a continuous DDPM on the {@code +-1} vectors and scores its thresholded samples.
     *
     * @param steps training steps on batches of 32
     * @param samplingSteps DDIM steps (network calls) per sample
     * @param seed the seed
     * @return the score
     */
    public static Score ddpm(int steps, int samplingSteps, long seed) {
        MlpDenoiser net = MlpDenoiser.of(BITS, 0, 64, 2, seed).timeScale(1f / 1000);
        Diffusion diffusion = new Diffusion(NoiseSchedule.cosine(1000), Diffusion.Prediction.VELOCITY).clipSample(1f);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int[] none = new int[32];
        Arrays.fill(none, -1);
        for (int s = 0; s < steps; s++) {
            int[][] d = data(32, r);
            float[][] x = new float[d.length][BITS];
            for (int b = 0; b < d.length; b++) {
                for (int i = 0; i < BITS; i++) {
                    x[b][i] = d[b][i];
                }
            }
            Var loss = diffusion.loss(net.bind(none), Nets.matrix(x), r);
            loss.backward();
            Nets.clip(net.parameters(), 1f);
            opt.step(net.parameters());
        }
        int[] all = new int[400];
        Arrays.fill(all, -1);
        Tensor noise = Nets.gaussian(all.length, BITS, new SplittableRandom(seed + 99));
        float[][] out = Nets.rows(diffusion.sample(net.bind(all), noise, samplingSteps, 0, r));
        int[][] samples = new int[out.length][BITS];
        for (int i = 0; i < out.length; i++) {
            for (int k = 0; k < BITS; k++) {
                samples[i][k] = out[i][k] >= 0 ? 1 : -1;
            }
        }
        return score("DDPM, " + samplingSteps + " steps", samples, seed);
    }

    /**
     * Scores samples against fresh data.
     *
     * @param method the method name
     * @param samples {@code +-1} samples
     * @param seed the seed for the reference data
     * @return the score
     */
    public static Score score(String method, int[][] samples, long seed) {
        Set<String> valid = new HashSet<>();
        for (int[] p : PATTERNS) {
            valid.add(Arrays.toString(p));
        }
        Set<String> hit = new HashSet<>();
        int ok = 0;
        for (int[] s : samples) {
            String key = Arrays.toString(s);
            if (valid.contains(key)) {
                ok++;
                hit.add(key);
            }
        }
        int[][] ref = data(samples.length, new SplittableRandom(seed + 7));
        double mmd = kernelMean(samples, samples) + kernelMean(ref, ref) - 2 * kernelMean(samples, ref);
        return new Score(method, (double) ok / samples.length, hit.size(), mmd);
    }

    private static double kernelMean(int[][] a, int[][] b) {
        double sum = 0;
        for (int[] x : a) {
            for (int[] y : b) {
                int h = 0;
                for (int i = 0; i < BITS; i++) {
                    h += x[i] != y[i] ? 1 : 0;
                }
                sum += Math.exp(-h / 4.0);
            }
        }
        return sum / ((double) a.length * b.length);
    }
}
