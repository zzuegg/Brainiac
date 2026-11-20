package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.tensor.Device;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The scheduler makes room as each kind's policy says, on a pretend GPU of 10 bytes. */
class GpuResidencyTest {

    @TempDir
    Path dir;

    private final AtomicLong used = new AtomicLong();
    private final Device device = Device.cpu();

    private GpuResidency residency(Path file) {
        GpuResidency r = new GpuResidency(file, d -> Optional.of(new Device.Memory(used.get(), 0, 0, 10)));
        r.timing(300, 10);
        return r;
    }

    private Path model(String name, long bytes) throws IOException {
        Path p = dir.resolve(name);
        try (RandomAccessFile f = new RandomAccessFile(p.toFile(), "rw")) {
            f.setLength(bytes);
        }
        return p;
    }

    // loads a pretend model of a file's size into a slot
    private void load(GpuSlot slot, Path model) throws IOException {
        long n = model.toFile().length();
        slot.lock();
        try {
            slot.load(model.getFileName().toString(), model, device, () -> false, () -> used.addAndGet(n),
                    () -> used.addAndGet(-n));
        } finally {
            slot.unlock();
        }
    }

    @Test
    void swapUnloadsTheIdleModelOfAnotherKind() throws IOException {
        GpuResidency r = residency(null);
        GpuSlot chat = new GpuSlot("chat", r);
        GpuSlot speech = new GpuSlot("speech", r);
        load(chat, model("coordinator", 6));
        load(speech, model("voice", 6));
        assertNull(chat.loadedModel());
        assertEquals("voice", speech.loadedModel());
        assertEquals(6, used.get());
    }

    @Test
    void aPinnedModelStaysAndTheUserIsAsked() throws IOException {
        GpuResidency r = residency(null);
        r.pin("chat", true);
        GpuSlot chat = new GpuSlot("chat", r);
        GpuSlot speech = new GpuSlot("speech", r);
        load(chat, model("coordinator", 6));
        GpuResidency.NoRoom e = assertThrows(GpuResidency.NoRoom.class, () -> load(speech, model("voice", 6)));
        assertEquals(GpuResidency.Policy.ASK, e.policy());
        assertEquals("coordinator", chat.loadedModel());
    }

    @Test
    void remoteAndAskGiveUpAtOnceAndSpillLoadsAnyway() throws Exception {
        GpuResidency r = residency(null);
        GpuSlot chat = new GpuSlot("chat", r);
        GpuSlot image = new GpuSlot("image", r);
        load(chat, model("coordinator", 6));
        Path pic = model("sdxl", 6);
        for (GpuResidency.Policy p : List.of(GpuResidency.Policy.REMOTE, GpuResidency.Policy.ASK)) {
            GpuResidency.NoRoom e = assertThrows(GpuResidency.NoRoom.class,
                    () -> ScopedValue.where(GpuResidency.POLICY, p).call(() -> {
                        load(image, pic);
                        return null;
                    }));
            assertEquals(p, e.policy());
        }
        ScopedValue.where(GpuResidency.POLICY, GpuResidency.Policy.SPILL).call(() -> {
            load(image, pic);
            return null;
        });
        assertEquals("coordinator", chat.loadedModel());
        assertEquals(12, used.get());
    }

    @Test
    void queueWaitsUntilTheRoomIsFreed() throws Exception {
        GpuResidency r = residency(null);
        r.change(Map.of("policies", Map.of("image", "queue")));
        GpuSlot chat = new GpuSlot("chat", r);
        GpuSlot image = new GpuSlot("image", r);
        load(chat, model("coordinator", 6));
        Path pic = model("sdxl", 6);
        assertThrows(GpuResidency.NoRoom.class, () -> load(image, pic));   // nobody frees it in time
        Thread freer = Thread.ofVirtual().start(() -> {
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                return;
            }
            r.unload("chat");
        });
        load(image, pic);
        freer.join();
        assertEquals("sdxl", image.loadedModel());
        assertNull(chat.loadedModel());
    }

    @Test
    void theChoicesAreKept() throws IOException {
        Path file = dir.resolve("gpu.json");
        GpuResidency r = residency(file);
        r.change(Map.of("policies", Map.of("speech", "remote"), "pinned", List.of("chat"), "idleMinutes", 15));
        GpuResidency again = residency(file);
        assertEquals(GpuResidency.Policy.REMOTE, again.policy("speech"));
        assertEquals(GpuResidency.Policy.SWAP, again.policy("image"));
        assertEquals(15, again.idleMinutes());
        assertEquals(List.of("chat"), again.toJson(device).get("pinned"));
        assertThrows(IllegalArgumentException.class, () -> r.change(Map.of("policies", Map.of("chat", "maybe"))));
    }

    @Test
    void anUnusedModelGoesAfterFiveMinutesUnlessTheUserChoseOtherwise() throws IOException {
        assertEquals(5, residency(null).idleMinutes());
        // a file from before the choice was kept holds 0, which nobody chose
        Path old = dir.resolve("old.json");
        java.nio.file.Files.writeString(old, "{\"idleMinutes\": 0}");
        assertEquals(5, residency(old).idleMinutes());
        Path chosen = dir.resolve("chosen.json");
        residency(chosen).change(Map.of("idleMinutes", 0));
        assertEquals(0, residency(chosen).idleMinutes());
    }

    @Test
    void aModelThatFindsTooLittleRoomItselfGetsTheOthersRoomAndTriesAgain() throws IOException {
        GpuResidency r = residency(null);
        GpuSlot chat = new GpuSlot("chat", r);
        GpuSlot image = new GpuSlot("image", r);
        load(chat, model("coordinator", 5));
        // the files say 5 fits in the 5 free; the model itself wants 1 more to work in
        Path flux = model("flux", 5);
        image.lock();
        try {
            image.load("flux", flux, device, () -> false, () -> {
                if (10 - used.get() < 6) {
                    throw new IllegalArgumentException("flux does not fit on the GPU");
                }
                return used.addAndGet(5);
            }, () -> used.addAndGet(-5));
        } finally {
            image.unlock();
        }
        assertNull(chat.loadedModel());
        assertEquals("flux", image.loadedModel());
    }
}
