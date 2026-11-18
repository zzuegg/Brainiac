package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;

import java.util.List;
import java.util.Map;

/**
 * The shape of a VITS model, as in a Hugging Face {@code config.json} ({@code VitsConfig}).
 *
 * @param vocabSize characters
 * @param hidden the text encoder's width
 * @param layers text encoder blocks
 * @param heads text encoder attention heads
 * @param window the relative-position window of attention
 * @param ffn the feed-forward width
 * @param ffnKernel the feed-forward convolution width
 * @param flowSize latent channels
 * @param layerNormEps the layer norm epsilon
 * @param stochasticDuration whether durations come from the stochastic predictor
 * @param speakers number of speakers
 * @param speakerEmbedding speaker embedding width, 0 for one speaker
 * @param upsampleInitialChannel the vocoder's first width
 * @param upsampleRates the vocoder's upsampling factors
 * @param upsampleKernels the vocoder's upsampling kernels
 * @param resblockKernels the vocoder's residual kernels
 * @param resblockDilations the vocoder's residual dilations
 * @param leakyReluSlope the vocoder's slope
 * @param dsChannels channels of the duration flows
 * @param dsLayers layers of the depth-separable convolutions
 * @param flowBins spline bins of the duration flows
 * @param tailBound spline range
 * @param durationKernel duration predictor kernel
 * @param durationFlows duration flows
 * @param durationFilterChannels the deterministic duration predictor's width
 * @param priorFlows coupling flows
 * @param priorWavenetLayers WaveNet layers per coupling flow
 * @param wavenetKernel WaveNet kernel
 * @param wavenetDilationRate WaveNet dilation growth
 * @param speakingRate the default speaking rate
 * @param noiseScale the latent noise
 * @param noiseScaleDuration the duration noise
 * @param sampleRate output samples per second
 */
public record VitsConfig(int vocabSize, int hidden, int layers, int heads, int window, int ffn, int ffnKernel,
        int flowSize, float layerNormEps, boolean stochasticDuration, int speakers, int speakerEmbedding,
        int upsampleInitialChannel, int[] upsampleRates, int[] upsampleKernels, int[] resblockKernels,
        int[][] resblockDilations, float leakyReluSlope, int dsChannels, int dsLayers, int flowBins, float tailBound,
        int durationKernel, int durationFlows, int durationFilterChannels, int priorFlows, int priorWavenetLayers,
        int wavenetKernel, int wavenetDilationRate, float speakingRate, float noiseScale, float noiseScaleDuration,
        int sampleRate) {

    /**
     * Reads a Hugging Face {@code config.json}, with {@code VitsConfig}'s defaults.
     *
     * @param c the parsed file
     * @return the shape
     * @throws IllegalArgumentException if it is not a VITS config
     */
    public static VitsConfig fromHuggingFace(Map<String, Object> c) {
        String type = Json.string(c, "model_type", "");
        if (!type.equals("vits")) {
            throw new IllegalArgumentException("not a VITS model: config.json has model_type '" + type + "'");
        }
        String act = Json.string(c, "hidden_act", "relu");
        if (!act.equals("relu")) {
            throw new IllegalArgumentException("VITS with hidden_act '" + act + "' is not supported (only relu)");
        }
        return new VitsConfig(i(c, "vocab_size", 38), i(c, "hidden_size", 192), i(c, "num_hidden_layers", 6),
                i(c, "num_attention_heads", 2), i(c, "window_size", 4), i(c, "ffn_dim", 768),
                i(c, "ffn_kernel_size", 3), i(c, "flow_size", 192), f(c, "layer_norm_eps", 1e-5),
                !(c.get("use_stochastic_duration_prediction") instanceof Boolean b) || b,
                i(c, "num_speakers", 1), i(c, "speaker_embedding_size", 0), i(c, "upsample_initial_channel", 512),
                ints(c, "upsample_rates", 8, 8, 2, 2), ints(c, "upsample_kernel_sizes", 16, 16, 4, 4),
                ints(c, "resblock_kernel_sizes", 3, 7, 11), dilations(c), f(c, "leaky_relu_slope", 0.1),
                i(c, "depth_separable_channels", 2), i(c, "depth_separable_num_layers", 3),
                i(c, "duration_predictor_flow_bins", 10), f(c, "duration_predictor_tail_bound", 5),
                i(c, "duration_predictor_kernel_size", 3), i(c, "duration_predictor_num_flows", 4),
                i(c, "duration_predictor_filter_channels", 256), i(c, "prior_encoder_num_flows", 4),
                i(c, "prior_encoder_num_wavenet_layers", 4), i(c, "wavenet_kernel_size", 5),
                i(c, "wavenet_dilation_rate", 1), f(c, "speaking_rate", 1), f(c, "noise_scale", 0.667),
                f(c, "noise_scale_duration", 0.8), i(c, "sampling_rate", 16_000));
    }

    private static int i(Map<String, Object> c, String key, int otherwise) {
        return (int) Json.integer(c, key, otherwise);
    }

    private static float f(Map<String, Object> c, String key, double otherwise) {
        return (float) Json.number(c, key, otherwise);
    }

    private static int[] ints(Map<String, Object> c, String key, int... otherwise) {
        if (!(c.get(key) instanceof List<?> l)) {
            return otherwise;
        }
        return l.stream().mapToInt(o -> ((Number) o).intValue()).toArray();
    }

    private static int[][] dilations(Map<String, Object> c) {
        if (!(c.get("resblock_dilation_sizes") instanceof List<?> l)) {
            return new int[][] {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
        }
        int[][] out = new int[l.size()][];
        for (int k = 0; k < out.length; k++) {
            out[k] = ((List<?>) l.get(k)).stream().mapToInt(o -> ((Number) o).intValue()).toArray();
        }
        return out;
    }

    /**
     * Output samples per latent frame: the product of the upsampling factors.
     *
     * @return the hop length
     */
    public int hop() {
        int h = 1;
        for (int r : upsampleRates) {
            h *= r;
        }
        return h;
    }
}
