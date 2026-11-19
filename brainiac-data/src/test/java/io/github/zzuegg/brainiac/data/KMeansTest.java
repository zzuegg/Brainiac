package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** Three separated blobs come back as three clusters with their centres. */
class KMeansTest {

    @Test
    void findsTheBlobs() {
        double[][] centres = {{0, 0}, {10, 0}, {0, 10}};
        SplittableRandom r = new SplittableRandom(3);
        double[][] x = new double[300][2];
        for (int i = 0; i < 300; i++) {
            x[i][0] = centres[i % 3][0] + r.nextDouble(-1, 1);
            x[i][1] = centres[i % 3][1] + r.nextDouble(-1, 1);
        }
        KMeans km = KMeans.fit(x, 3, 5, 100, 1);
        for (int i = 3; i < 300; i++) {
            assertEquals(km.labels()[i % 3], km.labels()[i], "row " + i);
        }
        assertNotEquals(km.labels()[0], km.labels()[1]);
        assertNotEquals(km.labels()[1], km.labels()[2]);
        assertNotEquals(km.labels()[0], km.labels()[2]);
        for (int c = 0; c < 3; c++) {
            double[] got = km.centroids()[km.labels()[c]];
            assertEquals(centres[c][0], got[0], 0.2);
            assertEquals(centres[c][1], got[1], 0.2);
        }
        assertEquals(km.labels()[0], km.predict(new double[][] {{0.3, -0.2}})[0]);
    }
}
