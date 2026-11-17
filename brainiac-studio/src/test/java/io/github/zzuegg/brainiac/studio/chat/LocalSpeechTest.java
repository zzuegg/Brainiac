package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.audio.Audio;
import io.github.zzuegg.brainiac.audio.Wav;
import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Speech models on this PC are listed with the others; a VITS model speaks through the studio. */
class LocalSpeechTest {

    // tiny random models from brainiac-audio's tests
    private static final Path FIXTURES = Path.of("../brainiac-audio/src/test/resources/io/github/zzuegg/brainiac/audio");

    @TempDir
    Path data;

    private Path models() throws IOException {
        copy(FIXTURES.resolve("vits"), data.resolve("models/tiny-tts"));
        copy(FIXTURES.resolve("whisper"), data.resolve("models/tiny-whisper"));
        return data.resolve("models");
    }

    private static void copy(Path from, Path to) throws IOException {
        try (Stream<Path> s = Files.walk(from)) {
            for (Path p : s.toList()) {
                Path t = to.resolve(from.relativize(p).toString());
                if (Files.isDirectory(p)) {
                    Files.createDirectories(t);
                } else {
                    Files.copy(p, t);
                }
            }
        }
    }

    private static Map<String, Object> model(Map<String, Object> d, String key) {
        return Json.array(d.get("models")).stream().map(Json::object).filter(x -> key.equals(x.get("key"))).findFirst()
                .orElseThrow(() -> new AssertionError(key + " not listed: " + d.get("models")));
    }

    @Test
    void aVitsModelSpeaksAndAWhisperModelIsListedToTranscribe() throws Exception {
        ChatService chat = ChatService.at(new Keys(data.resolve("keys.properties"), Map.of()), List.of(models()), Map.of(),
                data.resolve("outputs"));
        Map<String, Object> d = chat.describe(true);
        Map<String, Object> tts = model(d, "local/tiny-tts");
        assertEquals(List.of("speech"), tts.get("media"));
        assertEquals(List.of("wav"), Json.object(tts.get("speech")).get("formats"));
        assertEquals(List.of("transcribe"), model(d, "local/tiny-whisper").get("media"));
        assertEquals("local/tiny-tts", Json.object(d.get("auto")).get("speech"), "Auto speaks here when it can");

        ChatService.Generated g = chat.speech("auto", "Hello there.", "default", 1, "", "wav", null);
        assertEquals("audio/wav", g.media().getFirst().mime());
        Audio a = Wav.decode(Files.readAllBytes(g.files().getFirst()));
        assertTrue(a.seconds() > 0, "some audio came out");
        assertTrue(g.files().getFirst().getFileName().toString().endsWith(".wav"));

        // a recording that is not WAV is refused in words
        assertThrows(IllegalArgumentException.class, () -> chat.transcribe("local/tiny-whisper", new byte[64],
                new LocalSpeech.Listening(null, false, true, ""), s -> { }, m -> { }, () -> false));
        assertThrows(IOException.class, () -> chat.transcribe("local/tiny-tts", Wav.encode(a),
                new LocalSpeech.Listening(null, false, true, ""), s -> { }, m -> { }, () -> false),
                "a speaking model does not transcribe");
    }

    @Test
    void aWorkersLocalModelsOnlyChat() throws Exception {
        assertTrue(ChatService.localModels(List.of(models())).models().isEmpty());
    }
}
