package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.tensor.DType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Published models, when {@code -PtestModels=<folder>} names a folder holding
 * {@code mms-tts-eng}, {@code kokoro-82m} and {@code whisper-*} folders: speech made by each
 * text-to-speech model is transcribed back by Whisper.
 */
class PretrainedAudioTest {

    private static Path models() {
        String dir = System.getProperty("brainiac.test.models", "");
        Assumptions.assumeFalse(dir.isEmpty(), "no -PtestModels");
        return Path.of(dir);
    }

    private static String words(String s) {
        return s.toLowerCase(Locale.ROOT).replaceAll("[^a-z ]", "").replaceAll("\\s+", " ").strip();
    }

    @Test
    void speechRoundTrip() throws Exception {
        roundTrip("mms-tts-eng", SpeechOptions.defaults().seed(1),
                "The weather is nice today. Let us go for a walk in the park.");
    }

    @Test
    void kokoroRoundTrip() throws Exception {
        roundTrip("kokoro-82m", SpeechOptions.defaults().seed(1),
                "The weather is nice today. Let us go for a walk in the park.");
        roundTrip("kokoro-82m", SpeechOptions.defaults().voice("bm_george*0.6+af_bella*0.4").speed(1.2f),
                "On the 21st of March, tickets cost $5.50 each.");
    }

    private static void roundTrip(String model, SpeechOptions options, String text) throws Exception {
        Path models = models();
        Path tts = models.resolve(model);
        Assumptions.assumeTrue(Files.isDirectory(tts), "no " + model);
        Audio speech;
        try (TextToSpeech t = SpeechModels.textToSpeech(tts, TestDevice.get(), DType.FLOAT32)) {
            long t0 = System.nanoTime();
            List<Audio> chunks = new ArrayList<>();
            speech = t.synthesize(text, options, chunks::add);
            System.out.printf("%s: %.2f s of speech in %.2f s, %d chunks%n", model, speech.seconds(),
                    (System.nanoTime() - t0) / 1e9, chunks.size());
            assertEquals(text.split("[.!?](\\s|$)").length, chunks.size());
            assertTrue(speech.seconds() > 1.5 && speech.seconds() < 10, "seconds " + speech.seconds());
        }
        checkTranscripts(models, model, speech, text);
    }

    @Test
    void f5ClonesAKokoroVoice() throws Exception {
        Path models = models();
        Path kokoro = models.resolve("kokoro-82m");
        Path f5 = models.resolve("f5-tts");
        Assumptions.assumeTrue(Files.isDirectory(kokoro) && Files.isDirectory(f5), "no kokoro-82m and f5-tts");
        String refText = "Some call me nature, others call me mother nature.";
        Audio reference;
        try (TextToSpeech t = SpeechModels.textToSpeech(kokoro, TestDevice.get(), DType.FLOAT32)) {
            reference = t.synthesize(refText, SpeechOptions.defaults().voice("bf_emma"), null);
        }
        String text = "The weather is nice today. Let us go for a walk in the park.";
        Audio speech;
        try (TextToSpeech t = SpeechModels.textToSpeech(f5, TestDevice.get(), DType.FLOAT32)) {
            assertTrue(t.canClone());
            long t0 = System.nanoTime();
            List<Audio> chunks = new ArrayList<>();
            speech = t.synthesize(text, SpeechOptions.defaults().seed(1).reference(reference, refText), chunks::add);
            System.out.printf("f5-tts: %.2f s of speech in %.2f s, %d chunks%n", speech.seconds(),
                    (System.nanoTime() - t0) / 1e9, chunks.size());
            assertTrue(speech.seconds() > 1.5 && speech.seconds() < 10, "seconds " + speech.seconds());
        }
        checkTranscripts(models, "f5-tts", speech, text);
    }

    private static void checkTranscripts(Path models, String model, Audio speech, String text) throws Exception {
        Wav.write(speech, Path.of(System.getProperty("java.io.tmpdir"), "brainiac-" + model + ".wav"));
        try (var dirs = Files.list(models)) {
            for (Path dir : dirs.filter(Whisper::isWhisper).sorted().toList()) {
                try (SpeechToText stt = SpeechModels.speechToText(dir, TestDevice.get(), DType.FLOAT32)) {
                    long t0 = System.nanoTime();
                    List<Transcription.Segment> seen = new ArrayList<>();
                    Transcription r = stt.transcribe(speech, TranscriptionOptions.defaults(), seen::add);
                    System.out.printf("%s: %.2f s: [%s] %s%n", dir.getFileName(), (System.nanoTime() - t0) / 1e9,
                            r.language(), r.text());
                    for (Transcription.Segment s : r.segments()) {
                        System.out.printf("  %6.2f-%6.2f %s%n", s.start(), s.end(), s.text());
                    }
                    assertEquals("en", r.language());
                    assertEquals(seen, r.segments());
                    assertEquals(words(text), words(r.text()));
                }
            }
        }
    }
}
