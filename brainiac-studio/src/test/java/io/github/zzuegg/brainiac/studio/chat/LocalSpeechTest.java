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
    void aKokoroFolderListsItsEnglishVoicesWithoutLoading() throws Exception {
        Path k = data.resolve("models/kokoro-82m");
        Files.createDirectories(k.resolve("voices"));
        Files.writeString(k.resolve("config.json"), "{\"istftnet\": {}, \"plbert\": {}, \"vocab\": {}}");
        for (String v : List.of("am_adam", "af_heart", "bm_george", "jf_alpha", "readme")) {
            Files.writeString(k.resolve("voices/" + v + ".pt"), "not loaded");
        }
        ChatService chat = ChatService.at(new Keys(data.resolve("keys.properties"), Map.of()), List.of(data.resolve("models")),
                Map.of(), data.resolve("outputs"));
        Map<String, Object> m = model(chat.describe(true), "local/kokoro-82m");
        assertEquals(List.of("speech"), m.get("media"));
        Map<String, Object> speech = Json.object(m.get("speech"));
        assertEquals(List.of("af_heart", "am_adam", "bm_george"), speech.get("voices"), "English only, the default first");
        assertEquals("George (British male)", Json.object(speech.get("names")).get("bm_george"));
        assertEquals(true, speech.get("mix"));
    }

    @Test
    void anF5FolderSpeaksInAVoiceMadeFromARecording() throws Exception {
        copy(FIXTURES.resolve("f5"), data.resolve("models/tiny-f5"));
        copy(FIXTURES.resolve("vocos"), data.resolve("models/vocos"));
        ChatService chat = ChatService.at(new Keys(data.resolve("keys.properties"), Map.of()), List.of(data.resolve("models")),
                Map.of(), data.resolve("outputs"));
        Map<String, Object> m = model(chat.describe(true), "local/tiny-f5");
        assertEquals("F5-TTS voice cloning", m.get("detail"));
        Map<String, Object> speech = Json.object(m.get("speech"));
        assertEquals(true, speech.get("clone"));
        assertEquals(List.of(), speech.get("voices"), "no voices until one is made from a recording");
        assertTrue(Json.array(chat.describe(true).get("models")).stream().map(Json::object)
                .noneMatch(x -> "local/vocos".equals(x.get("key"))), "the vocoder is not a model of its own");

        float[] tone = new float[24_000 * 2];
        for (int i = 0; i < tone.length; i++) {
            tone[i] = (float) (0.2 * Math.sin(i * 2 * Math.PI * 220 / 24_000));
        }
        byte[] wav = Wav.encode(new Audio(tone, 24_000));
        assertThrows(IllegalArgumentException.class, () -> chat.saveVoice("local/tiny-f5", "Me", wav, ""),
                "without a transcript or a Whisper model to write it down");
        assertThrows(IllegalArgumentException.class, () -> chat.saveVoice("local/tiny-f5", "../up", wav, "hello"));
        assertEquals("hello there", chat.saveVoice("local/tiny-f5", "Me", wav, " hello there "));
        assertTrue(Files.isRegularFile(data.resolve("voices/tiny-f5/Me.wav")), "kept in the data folder");
        assertEquals(List.of("Me"), Json.object(model(chat.describe(true), "local/tiny-f5").get("speech")).get("voices"));

        ChatService.Generated g = chat.speech("local/tiny-f5", "Hi.", "Me", 1, "", "wav", null);
        assertTrue(Wav.decode(Files.readAllBytes(g.files().getFirst())).seconds() > 0, "it speaks in the new voice");

        chat.removeVoice("local/tiny-f5", "Me");
        assertEquals(List.of(), Json.object(model(chat.describe(true), "local/tiny-f5").get("speech")).get("voices"));
    }

    @Test
    void anOrpheusFolderSpeaksWithItsVoicesAndTagsAndIsNotAChatModel() throws Exception {
        Path o = Files.createDirectories(data.resolve("models/orpheus-3b"));
        Files.writeString(o.resolve("config.json"), "{\"model_type\": \"llama\", \"vocab_size\": 156940}");
        Files.writeString(o.resolve("tokenizer.json"), "{}");
        copy(FIXTURES.resolve("snac"), data.resolve("models/snac"));
        ChatService chat = ChatService.at(new Keys(data.resolve("keys.properties"), Map.of()), List.of(data.resolve("models")),
                Map.of(), data.resolve("outputs"));
        List<Map<String, Object>> all = Json.array(chat.describe(true).get("models")).stream().map(Json::object).toList();
        assertEquals(1, all.stream().filter(x -> "local/orpheus-3b".equals(x.get("key"))).count(), "listed once: " + all);
        Map<String, Object> m = model(chat.describe(true), "local/orpheus-3b");
        assertEquals(List.of("speech"), m.get("media"));
        assertEquals("Orpheus text to speech", m.get("detail"));
        Map<String, Object> speech = Json.object(m.get("speech"));
        assertEquals("tara", Json.array(speech.get("voices")).getFirst());
        assertEquals("Tara", Json.object(speech.get("names")).get("tara"));
        assertTrue(Json.array(speech.get("tags")).contains("laugh"), "tags without brackets: " + speech.get("tags"));
    }

    @Test
    void aCharacterSavedFromATakeBecomesAVoiceInFishAndChatterbox() throws Exception {
        copy(FIXTURES.resolve("fish"), data.resolve("models/tiny-fish"));
        copy(FIXTURES.resolve("chatterbox"), data.resolve("models/tiny-chatterbox"));
        ChatService chat = ChatService.at(new Keys(data.resolve("keys.properties"), Map.of()), List.of(data.resolve("models")),
                Map.of(), data.resolve("outputs"));
        Map<String, Object> fish = Json.object(model(chat.describe(true), "local/tiny-fish").get("speech"));
        assertEquals(Boolean.TRUE, fish.get("clone"));
        assertTrue(Json.array(fish.get("tags")).contains("[angry]"), "Fish's tags keep their brackets: " + fish.get("tags"));
        assertEquals(Boolean.TRUE, Json.object(model(chat.describe(true), "local/tiny-chatterbox").get("speech")).get("clone"));

        // what the page sends for each model when a take is saved as a character
        float[] tone = new float[3 * 24_000];
        for (int i = 0; i < tone.length; i++) {
            tone[i] = (float) (0.3 * Math.sin(2 * Math.PI * 140 * i / 24_000.0));
        }
        byte[] take = Wav.encode(new Audio(tone, 24_000));
        for (String m : List.of("local/tiny-fish", "local/tiny-chatterbox")) {
            chat.saveVoice(m, "Captain", take, "Aye, the sea gives and the sea takes.");
            assertTrue(Json.array(Json.object(model(chat.describe(true), m).get("speech")).get("voices")).contains("Captain"),
                    m + " lists the character");
        }
    }

    @Test
    void aWorkersLocalModelsOnlyChat() throws Exception {
        assertTrue(ChatService.localModels(List.of(models())).models().isEmpty());
    }
}
