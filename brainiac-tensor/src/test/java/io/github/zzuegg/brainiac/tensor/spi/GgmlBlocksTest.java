package io.github.zzuegg.brainiac.tensor.spi;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GgmlBlocksTest {

    /** A matrix of one byte per value, mapped from a file: each value is its byte. */
    private static GgmlBlocks mapped(MemorySegment data, int out, int in) {
        return new GgmlBlocks(data, "bytes", out, in, 1, 1, (blocks, count, dst, at) -> {
            for (int i = 0; i < count; i++) {
                dst[at + i] = blocks.get(java.lang.foreign.ValueLayout.JAVA_BYTE, i);
            }
        });
    }

    @Test
    void readsAheadAndLoadsWhatIsLeftOnce(@TempDir Path dir) throws IOException {
        int out = 3;
        int in = 1 << 20;   // three pieces of a megabyte each
        byte[] bytes = new byte[out * in];
        for (int i = 0; i < bytes.length; i++) {
            bytes[i] = (byte) (i * 31);
        }
        Path file = Files.write(dir.resolve("m.bin"), bytes);
        try (Arena arena = Arena.ofShared(); FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            MemorySegment data = ch.map(FileChannel.MapMode.READ_ONLY, 0, bytes.length, arena);
            GgmlBlocks a = mapped(data.asSlice(0, (long) in), 1, in);
            GgmlBlocks b = mapped(data.asSlice(in, 2L * in), 2, in);
            assertFalse(a.loaded());
            assertEquals(in, GgmlBlocks.readAhead(List.of(a)));
            assertEquals(0, GgmlBlocks.readAhead(List.of(a)), "asked for once while on its way");
            GgmlBlocks.loadAll(List.of(a, b));   // takes what the read-ahead left, and all of b
            assertTrue(b.loaded());
            long deadline = System.nanoTime() + 10_000_000_000L;
            while (!a.loaded() && System.nanoTime() < deadline) {
                Thread.yield();
            }
            assertTrue(a.loaded());
            assertEquals(0, GgmlBlocks.readAhead(List.of(a, b)), "in memory");
            float[] row = new float[in];
            b.decodeRow(1, row);
            assertEquals((byte) ((2 * in + 5) * 31), row[5]);
        }
    }
}
