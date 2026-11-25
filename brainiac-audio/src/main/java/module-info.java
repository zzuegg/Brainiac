/**
 * Speech models: WAV decoding, Whisper speech-to-text, and text-to-speech with VITS
 * (MMS-TTS), Kokoro, F5-TTS voice cloning (with Vocos) and Orpheus (with SNAC), loading
 * published weights, behind the {@link io.github.zzuegg.brainiac.audio.SpeechToText} and
 * {@link io.github.zzuegg.brainiac.audio.TextToSpeech} interfaces.
 */
module io.github.zzuegg.brainiac.audio {
    requires transitive io.github.zzuegg.brainiac.nn;
    requires transitive io.github.zzuegg.brainiac.io;
    requires transitive io.github.zzuegg.brainiac.llm;
    exports io.github.zzuegg.brainiac.audio;
    exports io.github.zzuegg.brainiac.audio.narration;
}
