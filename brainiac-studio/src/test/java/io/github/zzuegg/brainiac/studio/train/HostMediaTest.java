package io.github.zzuegg.brainiac.studio.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.audio.Transcription;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatProvider;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.Keys;
import io.github.zzuegg.brainiac.studio.chat.LocalImages;
import io.github.zzuegg.brainiac.studio.chat.LocalSpeech;
import io.github.zzuegg.brainiac.studio.chat.Media;
import io.github.zzuegg.brainiac.studio.chat.ModelInfo;
import io.github.zzuegg.brainiac.studio.server.WorkerServer;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** A worker host draws, speaks and writes down recordings with its models, and the studio uses them as its own. */
class HostMediaTest {

    private static final String TOKEN = "a-given-worker-token-0123456789abcdef";

    @TempDir
    Path studio;

    @TempDir
    Path lab;

    /** A worker's models that make small stand-ins and remember what they were asked. */
    static final class Maker extends HostChatTest.Echo {
        final List<Object> asked = new ArrayList<>();
        final AtomicBoolean stopped = new AtomicBoolean();

        @Override
        public List<ModelInfo> models() {
            return List.of(new ModelInfo("local", "sd-tiny", "SD tiny", List.of("image"), 0, "diffusers"),
                    new ModelInfo("local", "kokoro", "Kokoro", List.of("speech"), 0, "onnx"),
                    new ModelInfo("local", "whisper-tiny", "Whisper tiny", List.of("transcribe"), 0, "safetensors"),
                    new ModelInfo("local", "wan-tiny", "Wan tiny", List.of("video"), 0, "diffusers"));
        }

        @Override
        public boolean makesVideo() {
            return true;
        }

        @Override
        public Map<String, Object> videoOptions(String model) {
            return model.equals("wan-tiny") ? Map.of("kind", "WAN", "input", "TEXT_OR_IMAGE", "steps", 30, "fps", 16) : null;
        }

        @Override
        public io.github.zzuegg.brainiac.studio.chat.LocalVideos.Clip video(String model, String prompt,
                io.github.zzuegg.brainiac.studio.chat.LocalVideos.Settings settings,
                io.github.zzuegg.brainiac.studio.chat.LocalVideos.Listener listener) throws IOException {
            asked.add(List.of(model, prompt, settings));
            if (model.equals("missing")) {
                throw new IOException("no video model named missing");
            }
            for (int i = 1; i <= settings.steps(); i++) {
                if (!listener.step(i, settings.steps())) {
                    throw new CancellationException("stopped");
                }
            }
            byte[] first = settings.start() == null ? new byte[0] : settings.start().data();
            return new io.github.zzuegg.brainiac.studio.chat.LocalVideos.Clip(
                    new Media(new byte[] {(byte) 0x89, 'P', 'N', 'G', (byte) first.length}, "image/apng"),
                    settings.seed(), 832, 480, 33, 16);
        }

        @Override
        public boolean runsMedia() {
            return true;
        }

        @Override
        public List<LocalImages.Picture> draw(String model, String prompt, LocalImages.Settings settings,
                LocalImages.Listener listener) throws IOException {
            asked.add(List.of(model, prompt, settings));
            if (model.equals("missing")) {
                throw new IOException("no image model named missing");
            }
            for (int i = 1; i <= settings.steps(); i++) {
                try {
                    Thread.sleep(settings.steps() > 10 ? 20 : 0);   // a long drawing takes its time
                } catch (InterruptedException e) {
                    throw new IOException(e);
                }
                if (!listener.step(i, settings.steps(), null)) {
                    stopped.set(true);
                    throw new CancellationException("stopped");
                }
            }
            return List.of(new LocalImages.Picture(new Media(new byte[] {(byte) 0x89, 'P', 'N', 'G', 9}, "image/png"),
                    settings.seed()));
        }

        @Override
        public Media speak(String model, String text, String voice, double speed, BooleanSupplier stop) {
            asked.add(List.of(model, text, voice, speed));
            return new Media(("RIFF" + text).getBytes(java.nio.charset.StandardCharsets.UTF_8), "audio/wav");
        }

        @Override
        public Transcription transcribe(String model, byte[] wav, LocalSpeech.Listening listening,
                Consumer<Transcription.Segment> onSegment, BooleanSupplier stop) {
            asked.add(List.of(model, wav.length, listening));
            Transcription.Segment a = new Transcription.Segment(0, 1.5, "hello", -0.2, 0.01);
            Transcription.Segment b = new Transcription.Segment(1.5, 3, " there", -0.3, 0.02);
            onSegment.accept(a);
            onSegment.accept(b);
            return new Transcription("hello there", "en", List.of(a, b), 3);
        }

        @Override
        public Map<String, Object> voiceOptions(String model) {
            return model.equals("kokoro") ? Map.of("voices", List.of("af_heart", "bm_george"), "clone", true) : null;
        }
    }

    @Test
    void aHostDrawsSpeaksAndTranscribesForTheStudio() throws Exception {
        Maker maker = new Maker();
        try (WorkerServer worker = WorkerServer.start(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), lab,
                maker, TOKEN);
             TrainingService t = new TrainingService(studio)) {
            Hosts.Host h = t.hosts().add("lab", "127.0.0.1:" + worker.port(), TOKEN);
            ChatService chat = new ChatService(new Keys(studio.resolve("keys.properties"), Map.of()), List.of(),
                    studio.resolve("outputs"));
            chat.hosts(() -> t.hosts().list().stream()
                    .map(x -> (ChatProvider) new HostProvider(x, t.hosts().client(x))).toList());
            String host = "host-" + h.id() + "/";

            // the page learns the host's voices with its models
            Map<String, Object> d = chat.describe(true);
            Map<String, Object> kokoro = Json.array(d.get("models")).stream().map(Json::object)
                    .filter(m -> (host + "kokoro").equals(m.get("key"))).findFirst().orElseThrow();
            assertEquals(List.of("af_heart", "bm_george"), Json.object(kokoro.get("speech")).get("voices"));
            assertEquals(false, Json.object(kokoro.get("speech")).get("clone"), "a new voice is made on this PC only");

            // drawing: every step is heard, the picture and its seed come back and are saved
            List<int[]> steps = new ArrayList<>();
            LocalImages.Settings settings = new LocalImages.Settings("blurry", "1:1", 1, 3, 6.5, 42, "EULER");
            ChatService.Drawn drawn = chat.draw(host + "sd-tiny", "a red fox", settings, (step, total, preview) -> {
                steps.add(new int[] {step, total});
                return true;
            }, m -> assertEquals(host + "sd-tiny", m.key()));
            assertEquals(3, steps.size());
            assertArrayEquals(new int[] {3, 3}, steps.getLast());
            assertEquals(List.of(42L), drawn.seeds());
            assertEquals("image/png", drawn.generated().media().getFirst().mime());
            assertEquals(5, drawn.generated().media().getFirst().data().length);
            assertEquals(List.of("sd-tiny", "a red fox", settings), maker.asked.getLast());

            // stopping at a step stops the worker too
            assertThrows(CancellationException.class, () -> chat.draw(host + "sd-tiny", "a red fox",
                    new LocalImages.Settings("", "1:1", 1, 150, 7, 1, "EULER"), (step, total, preview) -> step < 2,
                    m -> { }));
            for (int i = 0; i < 100 && !maker.stopped.get(); i++) {
                Thread.sleep(50);
            }
            assertTrue(maker.stopped.get(), "the worker stops drawing when the studio stops listening");

            // a failure on the worker reaches the studio in the worker's words
            HostProvider direct = new HostProvider(h, t.hosts().client(h));
            IOException e = assertThrows(IOException.class,
                    () -> direct.draw("missing", "x", settings, (step, total, preview) -> true));
            assertEquals("no image model named missing", e.getMessage());

            // speaking
            ChatService.Generated said = chat.speech(host + "kokoro", "good morning", "bm_george", 1.25, "", "wav", null);
            assertEquals("RIFFgood morning", new String(said.media().getFirst().data(),
                    java.nio.charset.StandardCharsets.UTF_8));
            assertEquals(List.of("kokoro", "good morning", "bm_george", 1.25), maker.asked.getLast());

            // transcribing, by name and as Auto: this PC has no transcriber, so the host's is used
            byte[] wav = new byte[64];
            for (String model : List.of(host + "whisper-tiny", ChatService.AUTO)) {
                List<String> parts = new ArrayList<>();
                ChatService.Transcribed heard = chat.transcribe(model, wav,
                        new LocalSpeech.Listening("en", false, true, "names"), s -> parts.add(s.text()),
                        m -> assertEquals(host + "whisper-tiny", m.key()), () -> false);
                assertEquals(List.of("hello", " there"), parts);
                assertEquals("hello there", heard.transcription().text());
                assertEquals("en", heard.transcription().language());
                assertEquals(3.0, heard.transcription().seconds());
                assertEquals(1.5, heard.transcription().segments().getFirst().end());
                assertEquals(List.of("whisper-tiny", 64, new LocalSpeech.Listening("en", false, true, "names")),
                        maker.asked.getLast());
            }
        }
    }

    @Test
    void aHostMakesAVideoClipForTheStudio() throws Exception {
        Maker maker = new Maker();
        try (WorkerServer worker = WorkerServer.start(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), lab,
                maker, TOKEN);
             TrainingService t = new TrainingService(studio)) {
            Hosts.Host h = t.hosts().add("lab", "127.0.0.1:" + worker.port(), TOKEN);
            ChatService chat = new ChatService(new Keys(studio.resolve("keys.properties"), Map.of()), List.of(),
                    studio.resolve("outputs"));
            chat.hosts(() -> t.hosts().list().stream()
                    .map(x -> (ChatProvider) new HostProvider(x, t.hosts().client(x))).toList());
            String key = "host-" + h.id() + "/wan-tiny";

            Map<String, Object> wan = Json.array(chat.describe(true).get("models")).stream().map(Json::object)
                    .filter(m -> key.equals(m.get("key"))).findFirst().orElseThrow();
            assertEquals(true, wan.get("makesVideo"));
            assertEquals("TEXT_OR_IMAGE", chat.videoOptions(key).get("input"));
            assertEquals(Map.of(), chat.videoOptions("host-" + h.id() + "/sd-tiny"));

            // the settings and the first frame cross over, every step is heard, the clip comes back and is saved
            List<Integer> steps = new ArrayList<>();
            var settings = new io.github.zzuegg.brainiac.studio.chat.LocalVideos.Settings("blurry", "9:16", true, 2, 4, 5.5,
                    7, 0, 0, new Media(new byte[] {1, 2, 3}, "image/png"));
            ChatService.Filmed filmed = chat.video(key, "a fox runs", settings, (step, total) -> steps.add(step),
                    m -> assertEquals(key, m.key()));
            assertEquals(List.of(1, 2, 3, 4), steps);
            assertEquals(7, filmed.clip().seed());
            assertEquals(33, filmed.clip().frames());
            assertEquals(16, filmed.clip().fps());
            assertEquals(3, filmed.clip().media().data()[4], "the first frame reached the worker");
            assertTrue(java.nio.file.Files.size(filmed.file()) > 0);
            var got = (io.github.zzuegg.brainiac.studio.chat.LocalVideos.Settings) ((List<?>) maker.asked.getLast()).get(2);
            assertEquals("9:16", got.aspect());
            assertEquals("blurry", got.negativePrompt());
            assertTrue(got.small());
            assertEquals(5.5, got.guidance());

            IOException missing = assertThrows(IOException.class, () -> chat.video("host-" + h.id() + "/missing",
                    "a fox", settings, (step, total) -> true, m -> { }));
            assertTrue(missing.getMessage().contains("missing"), missing.getMessage());
        }
    }
}
