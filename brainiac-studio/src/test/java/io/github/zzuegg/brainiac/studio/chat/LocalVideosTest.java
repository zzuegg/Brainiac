package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.diffusion.VideoRequest;
import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Wan video models on this PC are listed with the others and make clips step by step. */
class LocalVideosTest {

    // a tiny random Wan 2.1 pipeline in diffusers layout, from brainiac-diffusion's tests
    private static final Path TINY = Path.of("../brainiac-diffusion/src/test/resources/tiny/wan");

    @TempDir
    Path data;

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

    @Test
    void aWanFolderIsAVideoModelThatMakesClipsHere() throws Exception {
        copy(TINY, data.resolve("models/tiny-wan"));
        ChatService chat = ChatService.at(new Keys(data.resolve("keys.properties"), Map.of()),
                List.of(data.resolve("models")), Map.of(), data.resolve("outputs"));
        Map<String, Object> d = chat.describe(true);
        List<Map<String, Object>> models = Json.array(d.get("models")).stream().map(Json::object)
                .filter(x -> "local/tiny-wan".equals(x.get("key"))).toList();
        assertEquals(1, models.size(), "listed once, as a video model and not as an image model");
        assertEquals(List.of("video"), models.get(0).get("media"));
        assertEquals("Wan video · diffusers", models.get(0).get("detail"));
        assertEquals("local/tiny-wan", Json.object(d.get("auto")).get("video"));
        Map<String, Object> o = chat.videoOptions("local/tiny-wan");
        assertEquals("WAN", o.get("kind"));
        assertEquals(16, o.get("fps"));

        List<int[]> steps = new ArrayList<>();
        AtomicReference<String> routed = new AtomicReference<>();
        // a size of its own, on the 16-pixel grid; half a second at 16 fps is 9 frames
        ChatService.Filmed f = chat.video("auto", "a red fox runs in snow",
                new LocalVideos.Settings("", "1:1", false, 0.5, 2, 5, 7, 30, 50), (step, total) -> {
                    steps.add(new int[] {step, total});
                    return true;
                }, x -> routed.set(x.key()));
        assertEquals("local/tiny-wan", routed.get());
        assertEquals(2, steps.size());
        assertEquals(32, f.clip().width());
        assertEquals(48, f.clip().height());
        assertEquals(9, f.clip().frames());
        assertEquals(7, f.clip().seed());
        byte[] png = Files.readAllBytes(f.file());
        assertEquals((byte) 0x89, png[0]);
        assertTrue(new String(png, 0, 200, java.nio.charset.StandardCharsets.ISO_8859_1).contains("acTL"),
                "an animated PNG");
        assertTrue(f.file().getFileName().toString().startsWith("video-"));

        assertThrows(CancellationException.class, () -> chat.video("local/tiny-wan", "stop me",
                new LocalVideos.Settings("", "1:1", false, 0.5, 3, 5, 1, 32, 32), (s, t) -> s < 2, x -> { }));
        assertThrows(IllegalArgumentException.class, () -> new LocalVideos.Settings("", "1:1", false, 30, 3, 5, 1));
    }

    @Test
    void settingsBecomeRequestsOnTheModelsGrid() {
        LocalVideos.Settings s = new LocalVideos.Settings("", "16:9", false, 2, 30, 5, 1);
        VideoRequest r = s.request("a fox", false, 16);
        assertEquals(832, r.width());
        assertEquals(480, r.height());
        assertEquals(33, r.frames());
        assertEquals(null, r.negativePrompt(), "Wan's own negative prompt when none is given");
        VideoRequest big = new LocalVideos.Settings("blur", "9:16", true, 5, 40, 5, 1).request("a fox", true, 24);
        assertEquals(352, big.width());
        assertEquals(640, big.height());
        assertEquals(121, big.frames());
        assertEquals("blur", big.negativePrompt());
        assertEquals("16:9", new LocalVideos.Settings("", "21:9", false, 1, 1, 1, 1).aspect());
        assertFalse(new LocalVideos.Settings("", "1:1", false, 1, 1, 1, 1).small());
        assertThrows(IllegalArgumentException.class, () -> new LocalVideos.Settings("", "1:1", false, 1, 1, 1, 1, 64, 0));
    }
}
