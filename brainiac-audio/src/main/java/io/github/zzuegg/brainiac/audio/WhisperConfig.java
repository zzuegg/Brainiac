package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;

import java.util.Map;

/**
 * The shape of a Whisper model, as in a Hugging Face {@code config.json}.
 *
 * @param mels mel bands of the input (80, or 128 for large-v3)
 * @param width the model width, {@code d_model}
 * @param encoderLayers encoder blocks
 * @param encoderHeads encoder attention heads
 * @param encoderFfn encoder feed-forward width
 * @param decoderLayers decoder blocks
 * @param decoderHeads decoder attention heads
 * @param decoderFfn decoder feed-forward width
 * @param vocabSize tokens
 * @param audioPositions encoder positions after the stride-2 convolution (1500)
 * @param textPositions decoder positions (448)
 */
public record WhisperConfig(int mels, int width, int encoderLayers, int encoderHeads, int encoderFfn,
        int decoderLayers, int decoderHeads, int decoderFfn, int vocabSize, int audioPositions, int textPositions) {

    /**
     * Reads a Hugging Face {@code config.json}.
     *
     * @param c the parsed file
     * @return the shape
     * @throws IllegalArgumentException if it is not a Whisper config
     */
    public static WhisperConfig fromHuggingFace(Map<String, Object> c) {
        String type = Json.string(c, "model_type", "");
        if (!type.equals("whisper")) {
            throw new IllegalArgumentException("not a Whisper model: config.json has model_type '" + type + "'");
        }
        int width = (int) Json.integer(c, "d_model");
        return new WhisperConfig((int) Json.integer(c, "num_mel_bins", 80), width,
                (int) Json.integer(c, "encoder_layers"), (int) Json.integer(c, "encoder_attention_heads"),
                (int) Json.integer(c, "encoder_ffn_dim", 4L * width),
                (int) Json.integer(c, "decoder_layers"), (int) Json.integer(c, "decoder_attention_heads"),
                (int) Json.integer(c, "decoder_ffn_dim", 4L * width),
                (int) Json.integer(c, "vocab_size"), (int) Json.integer(c, "max_source_positions", 1500),
                (int) Json.integer(c, "max_target_positions", 448));
    }

    /**
     * Whether the vocabulary is large-v3's (one more language, Cantonese).
     *
     * @return {@code true} for 51866 or more tokens
     */
    public boolean v3Vocabulary() {
        return vocabSize >= 51866;
    }

    /**
     * Whether the model knows languages other than English.
     *
     * @return {@code false} for the {@code .en} models
     */
    public boolean multilingual() {
        return vocabSize >= 51865;
    }
}
