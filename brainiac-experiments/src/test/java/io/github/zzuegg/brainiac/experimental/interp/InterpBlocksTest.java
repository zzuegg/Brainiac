package io.github.zzuegg.brainiac.experimental.interp;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments;
import io.github.zzuegg.brainiac.experimental.nn.LinearProbe;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class InterpBlocksTest {

    @Test
    void ckaIsInvariantToRotationAndMaxFlowIsExact() {
        SplittableRandom r = new SplittableRandom(1);
        float[][] x = new float[100][2];
        float[][] y = new float[100][2];
        double c = Math.cos(0.7);
        double s = Math.sin(0.7);
        for (int i = 0; i < x.length; i++) {
            x[i][0] = (float) r.nextGaussian();
            x[i][1] = (float) r.nextGaussian();
            y[i][0] = (float) (3 * (c * x[i][0] - s * x[i][1]));
            y[i][1] = (float) (3 * (s * x[i][0] + c * x[i][1]));
        }
        double cka = RepresentationSimilarity.linearCka(x, y);
        double[][] cap = new double[4][4];
        cap[0][1] = 3;
        cap[0][2] = 2;
        cap[1][3] = 2;
        cap[1][2] = 1;
        cap[2][3] = 3;
        double flow = AttentionRollout.maxFlow(cap, 0, 3);
        System.out.printf("CKA under rotation and scaling %.6f; max flow %.3f (exact 5)%n", cka, flow);
        assertEquals(1.0, cka, 1e-4);
        assertEquals(5.0, flow, 1e-9);
        assertTrue(GlyphDocuments.distinctGlyphs(), "every glyph of the document font is distinct");
    }

    @Test
    void leaceDefeatsLinearProbes() {
        SplittableRandom r = new SplittableRandom(2);
        int n = 400;
        float[][] x = new float[n][4];
        int[] z = new int[n];
        for (int i = 0; i < n; i++) {
            z[i] = r.nextInt(2);
            for (int j = 0; j < 4; j++) {
                x[i][j] = (float) (r.nextGaussian() + (j < 2 ? 1.5 * z[i] : 0));
            }
        }
        double before = LinearProbe.fit(x, z, 2, 200, 0.05, 1e-4).accuracy(x, z);
        float[][] erased = ConceptErasure.leace(x, z, 2).apply(x);
        double after = LinearProbe.fit(erased, z, 2, 200, 0.05, 1e-4).accuracy(erased, z);
        System.out.printf("LEACE: linear probe accuracy %.3f before, %.3f after%n", before, after);
        assertTrue(before > 0.8);
        assertTrue(after < 0.62);
    }
}
