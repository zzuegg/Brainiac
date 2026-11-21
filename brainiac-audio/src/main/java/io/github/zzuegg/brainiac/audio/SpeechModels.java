package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/**
 * Finds and loads speech models from Hugging Face folders.
 *
 * <pre>{@code
 * for (Path dir : Files.list(models).toList()) {
 *     SpeechModels.kind(dir).ifPresent(k -> System.out.println(dir + " is " + k));
 * }
 * SpeechToText stt = SpeechModels.speechToText(Path.of("whisper-base"), Device.cpu(), DType.FLOAT32);
 * TextToSpeech tts = SpeechModels.textToSpeech(Path.of("mms-tts-eng"), Device.cpu(), DType.FLOAT32);
 * }</pre>
 */
public final class SpeechModels {

    /** What a folder holds. */
    public enum Kind {
        /** A Whisper model: speech to text. */
        SPEECH_TO_TEXT,
        /**
         * A text-to-speech model: Kokoro, F5-TTS (with Vocos), Orpheus (with SNAC), Qwen3-TTS, Chatterbox, Fish Audio S2-Pro or VITS
         * (MMS-TTS and others).
         */
        TEXT_TO_SPEECH
    }

    private SpeechModels() {
    }

    /**
     * Returns what speech model a folder holds, from its {@code config.json}, without
     * reading the weights.
     *
     * @param dir a folder
     * @return the kind, or empty if it is not a speech model Brainiac can load
     */
    public static Optional<Kind> kind(Path dir) {
        if (Kokoro.isKokoro(dir) || F5Speech.isF5(dir) || OrpheusSpeech.isOrpheus(dir)
                || Qwen3Speech.isQwen3Tts(dir) || ChatterboxSpeech.isChatterbox(dir)
                || FishSpeech.isFish(dir)) {
            return Optional.of(Kind.TEXT_TO_SPEECH);
        }
        return switch (modelType(dir)) {
            case "whisper" -> Optional.of(Kind.SPEECH_TO_TEXT);
            case "vits" -> Optional.of(Kind.TEXT_TO_SPEECH);
            default -> Optional.empty();
        };
    }

    /** The {@code model_type} of a folder's {@code config.json}, or "" without one. */
    static String modelType(Path dir) {
        Path config = dir.resolve("config.json");
        if (!Files.isRegularFile(config)) {
            return "";
        }
        try {
            return Json.string(Json.object(Json.read(config)), "model_type", "");
        } catch (Exception e) {
            return "";
        }
    }

    /**
     * Loads a speech-to-text model.
     *
     * @param dir a Hugging Face Whisper folder
     * @param device where it runs
     * @param dtype the type of its weights ({@link DType#FLOAT32}, or a 16-bit type on a GPU)
     * @return the model, loaded until closed
     * @throws IllegalArgumentException if the folder holds no speech-to-text model
     */
    public static SpeechToText speechToText(Path dir, Device device, DType dtype) {
        if (kind(dir).orElse(null) != Kind.SPEECH_TO_TEXT) {
            throw new IllegalArgumentException(dir + " is not a speech-to-text model; expected a Hugging Face"
                    + " Whisper folder (config.json with model_type whisper), such as openai/whisper-base");
        }
        return WhisperTranscriber.load(dir, device, dtype);
    }

    /**
     * Loads a text-to-speech model.
     *
     * @param dir a Kokoro folder (hexgrad/Kokoro-82M), an F5-TTS folder (SWivid/F5-TTS, with
     *     charactr/vocos-mel-24khz in or next to it), an Orpheus folder
     *     (canopylabs/orpheus-3b-0.1-ft, with hubertsiuzdak/snac_24khz in or next to it), a Qwen3-TTS
     *     folder (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, -VoiceDesign or -Base), a Chatterbox folder
     *     (ResembleAI/chatterbox), a Fish Audio folder (fishaudio/s2-pro) or a Hugging Face VITS
     *     folder, such as facebook/mms-tts-eng
     * @param device where it runs
     * @param dtype the type of its weights
     * @return the model, loaded until closed
     * @throws IllegalArgumentException if the folder holds no text-to-speech model
     */
    public static TextToSpeech textToSpeech(Path dir, Device device, DType dtype) {
        if (kind(dir).orElse(null) != Kind.TEXT_TO_SPEECH) {
            throw new IllegalArgumentException(dir + " is not a text-to-speech model; expected a Kokoro folder"
                    + " (hexgrad/Kokoro-82M), F5-TTS, Orpheus, Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice),"
                    + " Chatterbox (ResembleAI/chatterbox), Fish Audio (fishaudio/s2-pro) or a"
                    + " Hugging Face VITS folder (config.json with model_type vits), such as facebook/mms-tts-eng");
        }
        if (Kokoro.isKokoro(dir)) {
            return KokoroSpeech.load(dir, device, dtype);
        }
        if (F5Speech.isF5(dir)) {
            return F5Speech.load(dir, device, dtype);
        }
        if (OrpheusSpeech.isOrpheus(dir)) {
            return OrpheusSpeech.load(dir, device, dtype);
        }
        if (Qwen3Speech.isQwen3Tts(dir)) {
            return Qwen3Speech.load(dir, device, dtype);
        }
        if (ChatterboxSpeech.isChatterbox(dir)) {
            return ChatterboxSpeech.load(dir, device, dtype);
        }
        if (FishSpeech.isFish(dir)) {
            return FishSpeech.load(dir, device, dtype);
        }
        return VitsSpeech.load(dir, device, dtype);
    }
}
