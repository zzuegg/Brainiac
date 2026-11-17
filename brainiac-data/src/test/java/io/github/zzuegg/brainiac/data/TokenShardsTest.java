package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TokenShardsTest {

    @Test
    void writesAndReadsAcrossShards(@TempDir Path dir) {
        try (TokenShards.Writer w = TokenShards.writer(dir, 7, 1000)) {
            w.add(new int[] {1, 2, 3}, 0);
            w.add(new int[] {4, 5, 6, 7, 8, 999}, 0);
            w.add(10, 11, 12, 13, 14);
            assertEquals(16, w.size());
        }
        assertTrue(Files.isRegularFile(dir.resolve("shard-00002.bin")));
        try (TokenShards s = TokenShards.open(dir)) {
            assertEquals(16, s.size());
            int[] all = new int[16];
            s.read(0, all);
            assertArrayEquals(new int[] {1, 2, 3, 0, 4, 5, 6, 7, 8, 999, 0, 10, 11, 12, 13, 14}, all);
            int[][] w = s.next(2, 4);
            assertArrayEquals(new int[] {1, 2, 3, 0, 4}, w[0]);
            assertArrayEquals(new int[] {4, 5, 6, 7, 8}, w[1]);   // starts at the last target
            assertEquals(8, s.position());
            s.seek(12);
            assertArrayEquals(new int[] {1, 2, 3, 0, 4}, s.next(1, 4)[0]);   // too few left: from the start
            int[][] r = s.random(50, 3, new SplittableRandom(1));
            for (int[] window : r) {
                assertEquals(4, window.length);
            }
            assertThrows(IllegalArgumentException.class, () -> s.next(1, 16));
        }
    }

    @Test
    void largeVocabulariesUse32Bits(@TempDir Path dir) {
        try (TokenShards.Writer w = TokenShards.writer(dir, 100, 200_000)) {
            w.add(150_000, 7);
        }
        try (TokenShards s = TokenShards.open(dir)) {
            assertEquals(150_000, s.token(0));
            assertEquals(7, s.token(1));
        }
        assertThrows(IllegalArgumentException.class, () -> TokenShards.writer(dir.resolve("x"), 10, 100).add(70_000));
    }

    @Test
    void readsNanoGptFiles(@TempDir Path dir) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        b.putShort((short) 50256).putShort((short) 1).putShort((short) 65535).putShort((short) 3);
        Path f = dir.resolve("train.bin");
        Files.write(f, b.array());
        try (TokenShards s = TokenShards.open(List.of(f), 16)) {
            assertEquals(4, s.size());
            assertEquals(50256, s.token(0));
            assertEquals(65535, s.token(2));
        }
    }
}
