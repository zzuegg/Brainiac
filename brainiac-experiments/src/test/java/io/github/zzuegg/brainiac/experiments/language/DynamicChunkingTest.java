package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DynamicChunkingTest {

    @Test
    void learnedChunksCompressAndPredictLikeFixedPatches() {
        List<DynamicChunking.Row> rows = DynamicChunking.run(110, 1);
        System.out.print(DynamicChunking.table(rows));
        DynamicChunking.Row perByte = rows.get(0);
        DynamicChunking.Row blt = rows.get(1);
        DynamicChunking.Row hnet = rows.get(2);
        DynamicChunking.Row bpe = rows.get(3);
        System.out.printf("bits/byte (bytes per main step): H-Net %.3f (%.2f) vs BLT entropy patches %.3f (%.2f) vs "
                + "BPE %.3f (%.2f) vs per byte %.3f (1.00); uniform = %.3f%n", hnet.bitsPerByte(), hnet.bytesPerStep(),
                blt.bitsPerByte(), blt.bytesPerStep(), bpe.bitsPerByte(), bpe.bytesPerStep(), perByte.bitsPerByte(),
                Math.log(9) / Math.log(2));
        assertTrue(hnet.bytesPerStep() > 1.5, "H-Net compresses");
        assertTrue(hnet.bitsPerByte() < Math.log(9) / Math.log(2) - 0.5, "H-Net learns the language");
        assertTrue(hnet.bitsPerByte() < blt.bitsPerByte() + 0.3);
    }
}
