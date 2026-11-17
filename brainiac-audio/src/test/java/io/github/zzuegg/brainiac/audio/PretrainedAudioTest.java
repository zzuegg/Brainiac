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
 * {@code mms-tts-eng} and {@code whisper-*} folders: speech made by MMS-TTS is transcribed
 * back by each Whisper.
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
        Path models = models();
        Path tts = models.resolve("mms-tts-eng");
        Assumptions.assumeTrue(Files.isDirectory(tts), "no mms-tts-eng");
        String text = "The weather is nice today. Let us go for a walk in the park.";
        Audio speech;
        try (TextToSpeech t = SpeechModels.textToSpeech(tts, TestDevice.get(), DType.FLOAT32)) {
            long t0 = System.nanoTime();
            List<Audio> chunks = new ArrayList<>();
            speech = t.synthesize(text, SpeechOptions.defaults().seed(1), chunks::add);
            System.out.printf("mms-tts-eng: %.2f s of speech in %.2f s, %d chunks%n", speech.seconds(),
                    (System.nanoTime() - t0) / 1e9, chunks.size());
            assertEquals(2, chunks.size());
            assertTrue(speech.seconds() > 2 && speech.seconds() < 10, "seconds " + speech.seconds());
        }
        Wav.write(speech, Path.of(System.getProperty("java.io.tmpdir"), "brainiac-tts.wav"));
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
