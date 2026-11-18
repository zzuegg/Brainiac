/**
 * Speech models: WAV decoding, Whisper speech-to-text and VITS (MMS-TTS) text-to-speech,
 * loading published Hugging Face weights, behind the {@link
 * io.github.zzuegg.brainiac.audio.SpeechToText} and {@link
 * io.github.zzuegg.brainiac.audio.TextToSpeech} interfaces.
 */
module io.github.zzuegg.brainiac.audio {
    requires transitive io.github.zzuegg.brainiac.nn;
    requires transitive io.github.zzuegg.brainiac.io;
    exports io.github.zzuegg.brainiac.audio;
}
