package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class HashGridTableSizeTest {

    @Test
    void collisionsHurtSmallTables() {
        double[] psnr = new double[4];
        int[] sizes = {8, 10, 12, 14};
        for (int i = 0; i < sizes.length; i++) {
            psnr[i] = SceneFields.fitImage(SceneFields.ImageNet.HASH_GRID, sizes[i], 48, 150, 1).psnr();
        }
        System.out.printf("hash-grid PSNR against table size 2^8/2^10/2^12/2^14: %.2f/%.2f/%.2f/%.2f dB%n", psnr[0],
                psnr[1], psnr[2], psnr[3]);
        assertTrue(psnr[2] > psnr[0] + 3, "collisions in small tables cost PSNR");
    }
}
