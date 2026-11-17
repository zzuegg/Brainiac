package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

/** Stable Diffusion models on this PC are listed with the others and draw step by step. */
class LocalImagesTest {

    // a tiny random Stable Diffusion 1.x model in diffusers layout, from brainiac-diffusion's tests
    private static final Path TINY = Path.of("../brainiac-diffusion/src/test/resources/tiny/sd");

    @TempDir
    Path data;

    private Path models() throws IOException {
        Path to = data.resolve("models/tiny-sd");
        try (Stream<Path> s = Files.walk(TINY)) {
            for (Path p : s.toList()) {
                Path t = to.resolve(TINY.relativize(p).toString());
                if (Files.isDirectory(p)) {
                    Files.createDirectories(t);
                } else {
                    Files.copy(p, t);
                }
            }
        }
        return data.resolve("models");
    }

    @Test
    void aStableDiffusionFolderIsAnImageModelThatDrawsHere() throws Exception {
        ChatService chat = ChatService.at(new Keys(data.resolve("keys.properties"), Map.of()), List.of(models()), Map.of(),
                data.resolve("outputs"));
        Map<String, Object> d = chat.describe(true);
        Map<String, Object> m = Json.array(d.get("models")).stream().map(Json::object)
                .filter(x -> "local/tiny-sd".equals(x.get("key"))).findFirst().orElseThrow();
        assertEquals(List.of("image"), m.get("media"));
        assertEquals("Stable Diffusion 1.x · diffusers", m.get("detail"));
        assertEquals("local/tiny-sd", Json.object(d.get("auto")).get("image"), "Auto draws here when it can");
        assertEquals("SD1", chat.drawOptions("local/tiny-sd").get("kind"));

        List<int[]> steps = new ArrayList<>();
        List<Integer> previews = new ArrayList<>();
        AtomicReference<String> routed = new AtomicReference<>();
        ChatService.Drawn drawn = chat.draw("auto", "a red square",
                new LocalImages.Settings("", "1:1", 2, 3, 7, 42, "EULER"), (step, total, preview) -> {
                    steps.add(new int[] {step, total});
                    if (preview != null) {
                        previews.add(preview.width());
                    }
                    return true;
                }, x -> routed.set(x.key()));
        assertEquals("local/tiny-sd", routed.get());
        assertEquals(6, steps.size(), "three steps for each of two pictures");
        assertArrayEquals(new int[] {6, 6}, steps.getLast());
        assertFalse(previews.isEmpty(), "rough previews come along the way");
        assertEquals(List.of(42L, 43L), drawn.seeds());
        assertEquals(2, drawn.generated().files().size());
        for (Path f : drawn.generated().files()) {
            byte[] png = Files.readAllBytes(f);
            assertEquals((byte) 0x89, png[0]);
            assertEquals('P', png[1]);
            assertTrue(f.getFileName().toString().startsWith("image-"));
        }

        // the same seed draws the same picture
        ChatService.Drawn again = chat.draw("local/tiny-sd", "a red square",
                new LocalImages.Settings("", "1:1", 1, 3, 7, 42, "EULER"), (s, t, p) -> true, x -> { });
        assertArrayEquals(drawn.generated().media().getFirst().data(), again.generated().media().getFirst().data());

        assertThrows(CancellationException.class, () -> chat.draw("local/tiny-sd", "stop me",
                new LocalImages.Settings("", "1:1", 1, 3, 7, 1, "EULER"), (s, t, p) -> s < 2, x -> { }));
        assertThrows(IllegalArgumentException.class, () -> new LocalImages.Settings("", "1:1", 5, 3, 7, 1, "EULER"));
    }

    @Test
    void aWorkersLocalModelsOnlyChat() throws Exception {
        assertTrue(ChatService.localModels(List.of(models())).models().isEmpty(),
                "a worker serves chat models; drawing is not on its API");
    }
}
