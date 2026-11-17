package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class PolarExpressTest {

    @Test
    void remezReproducesThePapersCoefficients() {
        double[][] c = PolarExpress.coefficients(8, 1e-3);
        System.out.printf("Polar Express coefficients: first (%.4f, %.4f, %.4f), last (%.4f, %.4f, %.4f)%n", c[0][0],
                c[0][1], c[0][2], c[7][0], c[7][1], c[7][2]);
        // the paper's first iteration for l = 1e-3 (before the safety factor)
        assertEquals(8.2872, c[0][0], 0.01);
        assertEquals(-23.5959, c[0][1], 0.03);
        assertEquals(17.3004, c[0][2], 0.03);
        // and the limit, the Newton-Schulz-like (15, -10, 3) / 8
        assertEquals(1.875, c[7][0], 1e-3);
        assertEquals(-1.25, c[7][1], 1e-3);
        assertEquals(0.375, c[7][2], 1e-3);
    }

    @Test
    void convergesToTheExactPolarFactorWhereNewtonSchulzStalls() {
        SplittableRandom r = new SplittableRandom(1);
        Tensor g = Tensor.normal(Shape.of(32, 64), 0, 1, r).matmul(Tensor.normal(Shape.of(64, 64), 0, 1, r));
        float pe5 = Orthogonalizer.error(PolarExpress.of(5).orthogonalize(g));
        float pe8 = Orthogonalizer.error(PolarExpress.of(8).orthogonalize(g));
        float ns5 = Orthogonalizer.error(Orthogonalizer.newtonSchulz(5).orthogonalize(g));
        float ns8 = Orthogonalizer.error(Orthogonalizer.newtonSchulz(8).orthogonalize(g));
        System.out.printf("Orthogonality error ||XX^T-I||/sqrt(k): Polar Express %.3f (5 it) %.4f (8 it) vs "
                + "Newton-Schulz %.3f / %.3f%n", pe5, pe8, ns5, ns8);
        assertTrue(pe8 < 0.05f && pe8 < 0.3f * ns8, "Polar Express converges, Newton-Schulz does not");
        // tall matrices are handled through the transpose
        Tensor tall = PolarExpress.of(8).orthogonalize(g.transpose());
        assertEquals(64, tall.shape().dim(0));
        assertTrue(Orthogonalizer.error(tall) < 0.05f);
    }
}
