package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class NerfEncodingTest {

    @Test
    void hashGridBeatsRawCoordinates() {
        SceneFields.Score raw = SceneFields.nerf(SceneFields.Encoding.NONE, true, 64, 120, 1);
        SceneFields.Score hash = SceneFields.nerf(SceneFields.Encoding.HASH_GRID, true, 64, 120, 1);
        System.out.printf("NeRF point encodings, held-out / train PSNR (hierarchical 16+16 samples): raw xyz"
                + " %.2f / %.2f dB, hash grid %.2f / %.2f dB (%.1f s vs %.1f s)%n", raw.psnr(), raw.trainPsnr(),
                hash.psnr(), hash.trainPsnr(), raw.trainSeconds(), hash.trainSeconds());
        assertTrue(hash.trainPsnr() > raw.trainPsnr(), "the hash grid fits more detail than raw coordinates");
    }
}
