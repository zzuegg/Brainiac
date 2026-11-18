package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.GroupNorm;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * The variational autoencoder of latent diffusion ({@code AutoencoderKL} in diffusers): an
 * encoder of {@link ResnetBlock}s and stride-2 convolutions down to a small latent image
 * with a mean and a log-variance per value, and a decoder back up, each with an
 * {@link ImageAttention} in the middle. With pretrained weights it is Stable Diffusion's VAE;
 * with fresh weights it trains with {@link #loss}.
 *
 * <p>As a layer it reconstructs its input {@code [batch, inChannels, h, w]} from the latent
 * mean (or, while training, from a sample).
 */
public final class AutoencoderKL implements Layer {

    /**
     * The shape of the autoencoder.
     *
     * @param inChannels image channels (3)
     * @param channels the width of each level, highest resolution first; each level after the
     *     first halves the size
     * @param layersPerBlock residual blocks per encoder level (the decoder has one more)
     * @param latentChannels the latent image's channels (4 for Stable Diffusion)
     * @param groups group-norm groups
     * @param epsilon the group norms' epsilon
     * @param scalingFactor what {@link #toLatents} multiplies by so latents have about unit variance
     * @param shiftFactor what {@link #toLatents} subtracts first (0 for Stable Diffusion 1 and XL)
     * @param quantConv whether 1x1 convolutions follow the encoder and precede the decoder
     * @param encoder whether the model has an encoder; without one it only decodes
     */
    public record Settings(int inChannels, int[] channels, int layersPerBlock, int latentChannels, int groups,
            float epsilon, float scalingFactor, float shiftFactor, boolean quantConv, boolean encoder) {

        /** Checks the values. */
        public Settings {
            if (channels.length < 1 || inChannels < 1 || layersPerBlock < 1 || latentChannels < 1) {
                throw new IllegalArgumentException("an autoencoder needs channels and at least one block per level");
            }
            channels = channels.clone();
        }

        /**
         * An autoencoder with two blocks per level, as many group-norm groups (up to 32) as
         * divide every width, and a scaling factor of 1.
         *
         * @param inChannels image channels
         * @param latentChannels latent channels
         * @param channels the width of each level
         * @return the settings
         */
        public static Settings of(int inChannels, int latentChannels, int... channels) {
            int groups = 32;
            for (int c : channels) {
                while (c % groups != 0) {
                    groups /= 2;
                }
            }
            return new Settings(inChannels, channels, 2, latentChannels, groups, 1e-6f, 1, 0, true, true);
        }

        /** @return how much smaller the latent image is per side */
        public int downscale() {
            return 1 << (channels.length - 1);
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Settings s && inChannels == s.inChannels && Arrays.equals(channels, s.channels)
                    && layersPerBlock == s.layersPerBlock && latentChannels == s.latentChannels
                    && groups == s.groups && Float.compare(epsilon, s.epsilon) == 0
                    && Float.compare(scalingFactor, s.scalingFactor) == 0
                    && Float.compare(shiftFactor, s.shiftFactor) == 0 && quantConv == s.quantConv
                    && encoder == s.encoder;
        }

        @Override
        public int hashCode() {
            return Objects.hash(inChannels, Arrays.hashCode(channels), layersPerBlock, latentChannels);
        }

        @Override
        public String toString() {
            return "AutoencoderKL.Settings[channels " + Arrays.toString(channels) + ", latent " + latentChannels
                    + ", scaling " + scalingFactor + "]";
        }
    }

    /**
     * The encoder's distribution over latents: a diagonal Gaussian.
     *
     * @param mean the means {@code [batch, latentChannels, h, w]}
     * @param logVariance the log-variances, the same shape, clamped to [-30, 20]
     */
    public record Posterior(Var mean, Var logVariance) {

        /**
         * Draws latents: {@code mean + exp(logVariance / 2) * noise}.
         *
         * @param random the source of noise
         * @return the latents
         */
        public Var sample(RandomGenerator random) {
            Shape s = mean.shape();
            float[] noise = new float[Math.toIntExact(s.size())];
            for (int i = 0; i < noise.length; i++) {
                noise[i] = (float) random.nextGaussian();
            }
            Tensor n = Tensor.of(s, noise, mean.value().device()).to(mean.value().dtype());
            return mean.add(logVariance.mul(0.5f).exp().mul(Var.of(n)));
        }

        /**
         * The KL divergence to a standard normal, summed over each sample's values and
         * averaged over the batch.
         *
         * @return a scalar
         */
        public Var kl() {
            int n = mean.shape().dim(0);
            Var perValue = mean.square().add(logVariance.exp()).sub(logVariance).sub(1).mul(0.5f);
            return perValue.sum().mul(1f / n);
        }
    }

    /** One level: residual blocks and the resampling convolution, or {@code null} on the last. */
    private record Level(List<ResnetBlock> resnets, Conv resample) {
    }

    /** One half of the autoencoder. */
    private record Coder(Conv convIn, List<Level> levels, ResnetBlock mid1, ImageAttention attention,
            ResnetBlock mid2, GroupNorm normOut, Conv convOut, Conv quant) {

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(convIn.parameters());
            for (Level l : levels) {
                for (ResnetBlock r : l.resnets()) {
                    p.addAll(r.parameters());
                }
                if (l.resample() != null) {
                    p.addAll(l.resample().parameters());
                }
            }
            p.addAll(mid1.parameters());
            p.addAll(attention.parameters());
            p.addAll(mid2.parameters());
            p.addAll(normOut.parameters());
            p.addAll(convOut.parameters());
            if (quant != null) {
                p.addAll(quant.parameters());
            }
            return p;
        }
    }

    private final Settings settings;
    private final Coder encoder;
    private final Coder decoder;
    private final SplittableRandom random;

    /**
     * Creates an autoencoder with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights and of latent samples while training
     */
    public AutoencoderKL(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        Settings s = settings;
        int[] ch = s.channels();
        int levels = ch.length;
        int latent = s.latentChannels();
        if (s.encoder()) {
            List<Level> down = new ArrayList<>();
            int width = ch[0];
            Conv in = Parts.conv3(s.inChannels(), ch[0], random);
            for (int i = 0; i < levels; i++) {
                List<ResnetBlock> resnets = new ArrayList<>();
                for (int j = 0; j < s.layersPerBlock(); j++) {
                    resnets.add(resnet(width, ch[i], random));
                    width = ch[i];
                }
                // stride 2 with one row and column of padding after, none before
                down.add(new Level(resnets, i < levels - 1 ? Parts.conv(width, width, 3, 2, 0, 1, random) : null));
            }
            encoder = new Coder(in, down, resnet(width, width, random), attention(width, random),
                    resnet(width, width, random), new GroupNorm(s.groups(), width, s.epsilon(), true),
                    Parts.conv3(width, 2 * latent, random), s.quantConv() ? Parts.conv1(2 * latent, 2 * latent, random)
                            : null);
        } else {
            encoder = null;
        }
        int width = ch[levels - 1];
        Conv in = Parts.conv3(latent, width, random);
        ResnetBlock mid1 = resnet(width, width, random);
        ImageAttention attention = attention(width, random);
        ResnetBlock mid2 = resnet(width, width, random);
        List<Level> up = new ArrayList<>();
        for (int i = 0; i < levels; i++) {
            int level = levels - 1 - i;
            List<ResnetBlock> resnets = new ArrayList<>();
            for (int j = 0; j <= s.layersPerBlock(); j++) {
                resnets.add(resnet(width, ch[level], random));
                width = ch[level];
            }
            up.add(new Level(resnets, i < levels - 1 ? Parts.conv3(width, width, random) : null));
        }
        decoder = new Coder(in, up, mid1, attention, mid2, new GroupNorm(s.groups(), width, s.epsilon(), true),
                Parts.conv3(width, s.inChannels(), random), s.quantConv() ? Parts.conv1(latent, latent, random) : null);
        this.random = random.split();
    }

    private ResnetBlock resnet(int in, int out, SplittableRandom random) {
        return new ResnetBlock(new ResnetBlock.Settings(in, out, 0, settings.groups(), settings.epsilon(), 0), random);
    }

    private ImageAttention attention(int width, SplittableRandom random) {
        return new ImageAttention(new ImageAttention.Settings(width, 1, settings.groups(), settings.epsilon()), random);
    }

    /**
     * Encodes images into the distribution of their latents.
     *
     * @param images {@code [batch, inChannels, h, w]}, h and w divisible by {@link Settings#downscale()}
     * @return the means and log-variances, in the model's number format
     * @throws IllegalStateException if the model has no encoder
     */
    public Posterior encode(Var images) {
        if (encoder == null) {
            throw new IllegalStateException(this + " has no encoder");
        }
        Var x = images.to(dtype());
        Var h = Parts.step(() -> encoder.convIn().forward(x));
        Parts.free(x, images);
        for (Level level : encoder.levels()) {
            for (ResnetBlock r : level.resnets()) {
                h = then(h, r::forward);
            }
            if (level.resample() != null) {
                h = then(h, level.resample()::forward);
            }
        }
        h = middle(encoder, h);
        Var moments = then(h, last -> {
            Var m = encoder.convOut().forward(Parts.silu(encoder.normOut().forward(last)));
            return encoder.quant() != null ? encoder.quant().forward(m) : m;
        });
        int l = settings.latentChannels();
        Var mean = Parts.step(() -> moments.slice(1, 0, l));
        Var logVariance = Parts.step(() -> moments.slice(1, l, 2 * l).clamp(-30f, 20f));
        Parts.free(moments);
        return new Posterior(mean, logVariance);
    }

    /**
     * Decodes latents into images.
     *
     * @param latents {@code [batch, latentChannels, h, w]}, as the encoder makes them (not scaled)
     * @return images {@code [batch, inChannels, h * downscale, w * downscale]}, typed like {@code latents}
     */
    public Var decode(Var latents) {
        Var z = Parts.step(() -> {
            Var s = latents.to(dtype());
            return decoder.quant() != null ? decoder.quant().forward(s) : s;
        });
        Var h = then(z, decoder.convIn()::forward, latents);
        h = middle(decoder, h);
        for (Level level : decoder.levels()) {
            for (ResnetBlock r : level.resnets()) {
                h = then(h, r::forward);
            }
            if (level.resample() != null) {
                h = then(h, in -> level.resample().forward(Parts.upsample2x(in)));
            }
        }
        return then(h, last -> decoder.convOut().forward(Parts.silu(decoder.normOut().forward(last)))
                .to(latents.value().dtype()));
    }

    /**
     * Encodes images into latents for a diffusion model: the mean, shifted and scaled by
     * the settings' factors.
     *
     * @param images {@code [batch, inChannels, h, w]}
     * @return the latents, float32
     */
    public Var toLatents(Var images) {
        Posterior p = encode(images);
        Var out = Parts.step(() -> p.mean().to(DType.FLOAT32).sub(settings.shiftFactor())
                .mul(settings.scalingFactor()));
        Parts.free(p.mean());
        Parts.free(p.logVariance());
        return out;
    }

    /**
     * Decodes latents from a diffusion model: undoes {@link #toLatents}'s scaling, then decodes.
     *
     * @param latents {@code [batch, latentChannels, h, w]}
     * @return the images, float32
     */
    public Var fromLatents(Var latents) {
        Var z = Parts.step(() -> latents.to(DType.FLOAT32).mul(1 / settings.scalingFactor())
                .add(settings.shiftFactor()));
        Var out = decode(z);
        Parts.free(z, latents);
        return out;
    }

    /**
     * The training loss for a batch of images: the mean squared error of the reconstruction
     * from sampled latents, plus {@code klWeight} times the KL divergence of the latents to a
     * standard normal.
     *
     * @param images {@code [batch, inChannels, h, w]}
     * @param klWeight the weight of the KL term, such as 1e-6
     * @param random the source of latent samples
     * @return a scalar
     */
    public Var loss(Var images, float klWeight, RandomGenerator random) {
        Posterior p = encode(images);
        Var reconstruction = decode(p.sample(random));
        Var error = reconstruction.sub(images).square().mean();
        return klWeight == 0 ? error : error.add(p.kl().mul(klWeight));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Posterior p = encode(input);
        return decode(training ? p.sample(random) : p.mean());
    }

    /** Runs {@code f} on {@code in} as one step and frees {@code in} unless it is one of {@code keep}. */
    private static Var then(Var in, java.util.function.UnaryOperator<Var> f, Var... keep) {
        Var out = Parts.step(() -> f.apply(in));
        Parts.free(in, keep);
        return out;
    }

    private static Var middle(Coder c, Var h) {
        h = then(h, c.mid1()::forward);
        h = then(h, c.attention()::forward);
        return then(h, c.mid2()::forward);
    }

    /**
     * Returns the sublayers in a fixed order, for loading weights: the encoder (if any) as
     * input convolution, each level's residual blocks and downsampling, middle residual
     * block, attention, middle residual block, group norm, output convolution and quant
     * convolution (if any); then the decoder as post-quant convolution (if any), input
     * convolution, the middle three, each level's residual blocks and upsampling convolution,
     * group norm and output convolution.
     *
     * @return the layers
     */
    public List<Layer> layers() {
        List<Layer> out = new ArrayList<>();
        if (encoder != null) {
            out.add(encoder.convIn());
            levels(out, encoder);
            out.addAll(List.of(encoder.mid1(), encoder.attention(), encoder.mid2(), encoder.normOut(),
                    encoder.convOut()));
            if (encoder.quant() != null) {
                out.add(encoder.quant());
            }
        }
        if (decoder.quant() != null) {
            out.add(decoder.quant());
        }
        out.addAll(List.of(decoder.convIn(), decoder.mid1(), decoder.attention(), decoder.mid2()));
        levels(out, decoder);
        out.addAll(List.of(decoder.normOut(), decoder.convOut()));
        return out;
    }

    private static void levels(List<Layer> out, Coder c) {
        for (Level l : c.levels()) {
            out.addAll(l.resnets());
            if (l.resample() != null) {
                out.add(l.resample());
            }
        }
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (encoder != null) {
            p.addAll(encoder.parameters());
        }
        p.addAll(decoder.parameters());
        return p;
    }

    /** @return the type the weights are held in */
    public DType dtype() {
        return decoder.convIn().weight().value().dtype();
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    @Override
    public String toString() {
        return "AutoencoderKL(channels " + Arrays.toString(settings.channels()) + ", latent "
                + settings.latentChannels() + (settings.encoder() ? "" : ", decoder only") + ")";
    }
}
