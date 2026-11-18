package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PatchDiscriminator;
import io.github.zzuegg.brainiac.experimental.nn.VqGan;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * VQ-GAN against VQ-VAE (wave 6, item 4.3) on 8x8 {@link ToyImages}, 16 tokens per image. The
 * same {@link VqGan} autoencoder is trained as a plain VQ-VAE with 16 codes (enough for every
 * 2x2 patch) and with 4 codes, and as a VQ-GAN with 4 codes (4x fewer): perceptual loss on the
 * frozen self-supervised
 * {@link RepaExperiment#encoder} plus a {@link PatchDiscriminator} hinge loss after a warm-up.
 * Scores on the 73 images: pixel mean squared error, the share of reconstructions that are
 * real shapes ({@link GenerativeScores#images}), sharpness (share of decisive pixels,
 * {@code |v| > 0.8}), and code usage (codes used and the code perplexity). Downstream MaskGIT
 * on the tokens was not run.
 */
@SuppressWarnings("exports")
public final class VqGanExperiment {
    private VqGanExperiment() {
    }

    private static void train(VqGan model, Optimizer dOpt, Model frozen, int iterations, long seed) {
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        model.initialize(Nets.matrix(PointClouds.images(256, r)));
        for (int it = 0; it < iterations; it++) {
            Tensor x = Nets.matrix(PointClouds.images(32, r));
            model.step(x, opt, dOpt);
            if (frozen != null) {
                for (Var p : frozen.parameters()) {
                    p.zeroGrad();
                }
            }
            if (it % 40 == 39 && it < iterations - 40) {
                model.restart(Nets.matrix(PointClouds.images(256, r)));
            }
        }
    }

    private static void score(Map<String, Double> out, String name, VqGan model, int codes) {
        float[][] images = LatentDiffusion.images();
        Tensor x = Nets.matrix(images);
        float[] rec = model.reconstruct(x).toArray();
        double mse = 0;
        int sharp = 0;
        for (int i = 0; i < images.length; i++) {
            for (int p = 0; p < ToyImages.PIXELS; p++) {
                double v = rec[i * ToyImages.PIXELS + p];
                mse += (v - images[i][p]) * (v - images[i][p]);
                sharp += Math.abs(v) > 0.8 ? 1 : 0;
            }
        }
        int total = images.length * ToyImages.PIXELS;
        int[] ids = model.codes(x);
        int[] counts = new int[codes];
        for (int id : ids) {
            counts[id]++;
        }
        int used = 0;
        double entropy = 0;
        for (int c : counts) {
            if (c > 0) {
                used++;
                double p = (double) c / ids.length;
                entropy -= p * Math.log(p);
            }
        }
        out.put(name + " MSE", mse / total);
        out.put(name + " real", GenerativeScores.images(Nets.rows(model.reconstruct(x))).realShare());
        out.put(name + " sharp", (double) sharp / total);
        out.put(name + " codes used", (double) used);
        out.put(name + " perplexity", Math.exp(entropy));
    }

    /**
     * Trains the three autoencoders.
     *
     * @param iterations steps of 32 images
     * @param seed the seed
     * @return per model ({@code VQ-VAE 16}, {@code VQ-VAE 4}, {@code VQ-GAN 4}) its scores
     */
    public static Map<String, Double> run(int iterations, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        VqGan big = VqGan.of(ToyImages.SIDE, 16, 8, 16, seed);
        train(big, null, null, iterations, seed);
        score(out, "VQ-VAE 16", big, 16);
        VqGan small = VqGan.of(ToyImages.SIDE, 4, 8, 16, seed);
        train(small, null, null, iterations, seed);
        score(out, "VQ-VAE 4", small, 4);
        Model encoder = RepaExperiment.encoder(300, seed);
        PatchDiscriminator d = PatchDiscriminator.of(1, 16, seed + 2);
        VqGan gan = VqGan.of(ToyImages.SIDE, 4, 8, 16, seed).perceptual(x -> encoder.forward(x, false), 0.1f)
                .adversarial(d, 0.1f, iterations / 3);
        train(gan, Optimizer.adam(2e-3f), encoder, iterations, seed);
        score(out, "VQ-GAN 4", gan, 4);
        return out;
    }
}
