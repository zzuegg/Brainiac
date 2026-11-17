package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.VariationalBottleneck;
import io.github.zzuegg.brainiac.experimental.nn.VectorQuantizer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A variational autoencoder (with free bits) and a VQ-VAE side by side on the 8x8
 * {@link ToyImages}. The VAE squeezes an image into a few Gaussian latents and can be
 * sampled from the prior; the VQ-VAE squeezes it into a handful of discrete codes, the
 * representation that token-based generators (MaskGIT, audio LMs) build on.
 */
public final class ImageAutoencoders {
    private ImageAutoencoders() {
    }

    /** A trained VAE and what it reports. */
    public static final class Vae {
        private final Model encoder;
        private final Model decoder;
        private final VariationalBottleneck bottleneck;
        private final int latent;
        private final float freeBits;
        private final Optimizer optimizer = Optimizer.adam(3e-3f);
        private final SplittableRandom random;

        Vae(int latent, float freeBits, long seed) {
            this.latent = latent;
            this.freeBits = freeBits;
            this.random = new SplittableRandom(seed);
            encoder = Model.sequential().input(ToyImages.PIXELS).dense(96, Activation.SILU).dense(2 * latent)
                    .seed(seed).build();
            decoder = Model.sequential().input(latent).dense(96, Activation.SILU).dense(ToyImages.PIXELS)
                    .seed(seed + 1).build();
            bottleneck = VariationalBottleneck.of(latent, seed);
        }

        /**
         * One step on the negative ELBO: squared reconstruction error plus the KL with free bits.
         *
         * @param images a batch
         * @return {@code {reconstruction, plain KL}}
         */
        public float[] trainStep(float[][] images) {
            Tensor x = Nets.matrix(images);
            Var stats = encoder.forward(Var.of(x), true);
            Var recon = decoder.forward(bottleneck.forward(stats, true), true).tanh();
            // a Gaussian likelihood with variance 0.1 on pixels in [-1, 1]
            Var rec = recon.sub(x).square().sum(1, false).mean().mul(5f);
            Var loss = rec.add(VariationalBottleneck.kl(stats, freeBits));
            float plainKl = VariationalBottleneck.kl(Var.of(stats.value()), 0).value().item();
            float r = rec.value().item();
            loss.backward();
            List<Var> params = new ArrayList<>(encoder.parameters());
            params.addAll(decoder.parameters());
            optimizer.step(params);
            return new float[] {r, plainKl};
        }

        /**
         * Counts latents the encoder actually uses: plain KL above 0.1 nats.
         *
         * @param images a probe batch
         * @return the number of active latents
         */
        public int activeLatents(float[][] images) {
            float[] kl = VariationalBottleneck.klPerDimension(encoder.forward(Var.of(Nets.matrix(images))).value());
            int n = 0;
            for (float k : kl) {
                n += k > 0.1f ? 1 : 0;
            }
            return n;
        }

        /**
         * Decodes samples of the standard normal prior.
         *
         * @param n how many
         * @param seed the seed
         * @return images
         */
        public float[][] sample(int n, long seed) {
            Tensor z = Nets.gaussian(n, latent, new SplittableRandom(seed));
            return Nets.rows(decoder.forward(Var.of(z)).value().tanh());
        }
    }

    /** A trained VQ-VAE. */
    public static final class VqVae {
        private final Model encoder;
        private final Model decoder;
        private final VectorQuantizer quantizer;
        private final Optimizer optimizer = Optimizer.adam(3e-3f);

        VqVae(int codes, int dim, long seed) {
            encoder = Model.sequential().input(ToyImages.PIXELS).dense(96, Activation.SILU).dense(dim).seed(seed)
                    .build();
            decoder = Model.sequential().input(dim).dense(96, Activation.SILU).dense(ToyImages.PIXELS)
                    .seed(seed + 1).build();
            quantizer = VectorQuantizer.of(codes, dim, 0.25f, seed);
        }

        /**
         * One step: reconstruction through the straight-through quantizer plus the
         * codebook and commitment terms.
         *
         * @param images a batch
         * @return the reconstruction loss
         */
        public float trainStep(float[][] images) {
            Tensor x = Nets.matrix(images);
            VectorQuantizer.Result q = quantizer.quantize(encoder.forward(Var.of(x), true));
            Var rec = decoder.forward(q.output(), true).tanh().sub(x).square().mean();
            float r = rec.value().item();
            rec.add(q.loss()).backward();
            List<Var> params = new ArrayList<>(encoder.parameters());
            params.addAll(decoder.parameters());
            params.addAll(quantizer.parameters());
            optimizer.step(params);
            return r;
        }

        /**
         * Seeds the codebook from encoder outputs (k-means++).
         *
         * @param images a batch
         */
        public void initialize(float[][] images) {
            quantizer.initialize(encoder.forward(Var.of(Nets.matrix(images))).value());
        }

        /**
         * Re-seeds unused codes.
         *
         * @param images a batch
         * @return the codes restarted
         */
        public int restart(float[][] images) {
            return quantizer.restartDeadCodes(encoder.forward(Var.of(Nets.matrix(images))).value());
        }

        /**
         * Encodes images to code ids.
         *
         * @param images images
         * @return one code per image
         */
        public int[] encode(float[][] images) {
            return quantizer.nearest(encoder.forward(Var.of(Nets.matrix(images))).value());
        }

        /**
         * Decodes code ids.
         *
         * @param codes codes
         * @return images
         */
        public float[][] decode(int[] codes) {
            return Nets.rows(decoder.forward(quantizer.lookup(codes)).value().tanh());
        }
    }

    /**
     * Creates an untrained VAE.
     *
     * @param latent the latent size
     * @param freeBits the free-bits floor per latent, in nats
     * @param seed the seed
     * @return the VAE
     */
    public static Vae vae(int latent, float freeBits, long seed) {
        return new Vae(latent, freeBits, seed);
    }

    /**
     * Creates an untrained VQ-VAE with one code per image.
     *
     * @param codes the codebook size
     * @param dim the code vector size
     * @param seed the seed
     * @return the VQ-VAE
     */
    public static VqVae vqVae(int codes, int dim, long seed) {
        return new VqVae(codes, dim, seed);
    }
}
