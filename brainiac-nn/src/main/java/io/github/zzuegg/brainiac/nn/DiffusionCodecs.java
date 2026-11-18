package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.nn.diffusion.AutoencoderKL;
import io.github.zzuegg.brainiac.nn.diffusion.ClipTextEncoder;
import io.github.zzuegg.brainiac.nn.diffusion.FluxTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.ImageAttention;
import io.github.zzuegg.brainiac.nn.diffusion.ResnetBlock;
import io.github.zzuegg.brainiac.nn.diffusion.SpatialTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.T5Encoder;
import io.github.zzuegg.brainiac.nn.diffusion.TimestepEmbedding;
import io.github.zzuegg.brainiac.nn.diffusion.UNet;

import java.util.SplittableRandom;

/** Model-file names and settings of the diffusion layers in {@code nn.diffusion}. */
final class DiffusionCodecs {

    private DiffusionCodecs() {
    }

    static void register() {
        Registry.simple("timestep_embedding", TimestepEmbedding.class, l -> {
            TimestepEmbedding.Settings s = l.settings();
            return Config.builder().put("frequencies", s.frequencies()).put("features", s.features())
                    .put("flipSinToCos", s.flipSinToCos()).put("freqShift", s.freqShift()).build();
        }, c -> new TimestepEmbedding(new TimestepEmbedding.Settings(c.getInt("frequencies"), c.getInt("features"),
                c.getBoolean("flipSinToCos"), c.getInt("freqShift")), new SplittableRandom()));
        Registry.simple("resnet_block", ResnetBlock.class, l -> {
            ResnetBlock.Settings s = l.settings();
            return Config.builder().put("channels", s.channels()).put("filters", s.filters())
                    .put("timeFeatures", s.timeFeatures()).put("groups", s.groups()).put("epsilon", s.epsilon())
                    .put("dropout", s.dropout()).build();
        }, c -> new ResnetBlock(new ResnetBlock.Settings(c.getInt("channels"), c.getInt("filters"),
                c.getInt("timeFeatures"), c.getInt("groups"), c.getFloat("epsilon"), c.getFloat("dropout")),
                new SplittableRandom()));
        Registry.simple("spatial_transformer", SpatialTransformer.class, l -> {
            SpatialTransformer.Settings s = l.settings();
            return Config.builder().put("channels", s.channels()).put("heads", s.heads()).put("depth", s.depth())
                    .put("contextFeatures", s.contextFeatures()).put("groups", s.groups())
                    .put("linearProjection", s.linearProjection()).build();
        }, c -> new SpatialTransformer(new SpatialTransformer.Settings(c.getInt("channels"), c.getInt("heads"),
                c.getInt("depth"), c.getInt("contextFeatures"), c.getInt("groups"), c.getBoolean("linearProjection")),
                new SplittableRandom()));
        Registry.simple("image_attention", ImageAttention.class, l -> {
            ImageAttention.Settings s = l.settings();
            return Config.builder().put("channels", s.channels()).put("heads", s.heads()).put("groups", s.groups())
                    .put("epsilon", s.epsilon()).build();
        }, c -> new ImageAttention(new ImageAttention.Settings(c.getInt("channels"), c.getInt("heads"),
                c.getInt("groups"), c.getFloat("epsilon")), new SplittableRandom()));
        Registry.simple("unet", UNet.class, l -> {
            UNet.Settings s = l.settings();
            return Config.builder().put("inChannels", s.inChannels()).put("outChannels", s.outChannels())
                    .put("channels", s.channels()).put("attention", ints(s.attention()))
                    .put("layersPerBlock", s.layersPerBlock()).put("heads", s.heads()).put("depth", s.depth())
                    .put("contextFeatures", s.contextFeatures()).put("linearProjection", s.linearProjection())
                    .put("groups", s.groups()).put("epsilon", s.epsilon()).put("addTimeFeatures", s.addTimeFeatures())
                    .put("pooledFeatures", s.pooledFeatures()).put("timeIds", s.timeIds())
                    .put("flipSinToCos", s.flipSinToCos()).put("freqShift", s.freqShift()).build();
        }, c -> new UNet(new UNet.Settings(c.getInt("inChannels"), c.getInt("outChannels"), c.getInts("channels"),
                booleans(c.getInts("attention")), c.getInt("layersPerBlock"), c.getInts("heads"), c.getInts("depth"),
                c.getInt("contextFeatures"), c.getBoolean("linearProjection"), c.getInt("groups"),
                c.getFloat("epsilon"), c.getInt("addTimeFeatures"), c.getInt("pooledFeatures"), c.getInt("timeIds"),
                c.getBoolean("flipSinToCos"), c.getInt("freqShift")), new SplittableRandom()));
        Registry.simple("autoencoder_kl", AutoencoderKL.class, l -> {
            AutoencoderKL.Settings s = l.settings();
            return Config.builder().put("inChannels", s.inChannels()).put("channels", s.channels())
                    .put("layersPerBlock", s.layersPerBlock()).put("latentChannels", s.latentChannels())
                    .put("groups", s.groups()).put("epsilon", s.epsilon()).put("scalingFactor", s.scalingFactor())
                    .put("shiftFactor", s.shiftFactor()).put("quantConv", s.quantConv()).put("encoder", s.encoder())
                    .build();
        }, c -> new AutoencoderKL(new AutoencoderKL.Settings(c.getInt("inChannels"), c.getInts("channels"),
                c.getInt("layersPerBlock"), c.getInt("latentChannels"), c.getInt("groups"), c.getFloat("epsilon"),
                c.getFloat("scalingFactor"), c.getFloat("shiftFactor"), c.getBoolean("quantConv"),
                c.getBoolean("encoder")), new SplittableRandom()));
        Registry.simple("clip_text_encoder", ClipTextEncoder.class, l -> {
            ClipTextEncoder.Settings s = l.settings();
            return Config.builder().put("vocabulary", s.vocabulary()).put("features", s.features())
                    .put("layers", s.layers()).put("heads", s.heads()).put("ffnSize", s.ffnSize())
                    .put("positions", s.positions()).put("activation", s.activation()).put("epsilon", s.epsilon())
                    .put("endToken", s.endToken()).put("projection", s.projection()).build();
        }, c -> new ClipTextEncoder(new ClipTextEncoder.Settings(c.getInt("vocabulary"), c.getInt("features"),
                c.getInt("layers"), c.getInt("heads"), c.getInt("ffnSize"), c.getInt("positions"),
                c.getActivation("activation"), c.getFloat("epsilon"), c.getInt("endToken"), c.getInt("projection")),
                new SplittableRandom()));
        Registry.simple("flux_transformer", FluxTransformer.class, l -> {
            FluxTransformer.Settings s = l.settings();
            return Config.builder().put("latentChannels", s.latentChannels()).put("patch", s.patch())
                    .put("heads", s.heads()).put("headSize", s.headSize()).put("jointLayers", s.jointLayers())
                    .put("singleLayers", s.singleLayers()).put("contextFeatures", s.contextFeatures())
                    .put("pooledFeatures", s.pooledFeatures()).put("guidance", s.guidance()).put("axes", s.axes())
                    .put("theta", s.theta()).put("mlpRatio", s.mlpRatio()).build();
        }, c -> new FluxTransformer(new FluxTransformer.Settings(c.getInt("latentChannels"), c.getInt("patch"),
                c.getInt("heads"), c.getInt("headSize"), c.getInt("jointLayers"), c.getInt("singleLayers"),
                c.getInt("contextFeatures"), c.getInt("pooledFeatures"), c.getBoolean("guidance"), c.getInts("axes"),
                c.getFloat("theta"), c.getFloat("mlpRatio")), new SplittableRandom()));
        Registry.simple("t5_encoder", T5Encoder.class, l -> {
            T5Encoder.Settings s = l.settings();
            return Config.builder().put("vocabulary", s.vocabulary()).put("features", s.features())
                    .put("layers", s.layers()).put("heads", s.heads()).put("headSize", s.headSize())
                    .put("ffnSize", s.ffnSize()).put("buckets", s.buckets()).put("maxDistance", s.maxDistance())
                    .put("epsilon", s.epsilon()).put("gated", s.gated()).build();
        }, c -> new T5Encoder(new T5Encoder.Settings(c.getInt("vocabulary"), c.getInt("features"), c.getInt("layers"),
                c.getInt("heads"), c.getInt("headSize"), c.getInt("ffnSize"), c.getInt("buckets"),
                c.getInt("maxDistance"), c.getFloat("epsilon"), c.getBoolean("gated")), new SplittableRandom()));
    }

    private static int[] ints(boolean[] b) {
        int[] out = new int[b.length];
        for (int i = 0; i < b.length; i++) {
            out[i] = b[i] ? 1 : 0;
        }
        return out;
    }

    private static boolean[] booleans(int[] v) {
        boolean[] out = new boolean[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] != 0;
        }
        return out;
    }
}
