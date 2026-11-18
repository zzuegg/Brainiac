package io.github.zzuegg.brainiac.experimental.parse;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.SplittableRandom;

class TreeDecoderTest {

    private static double score(double[][] s, int[] head) {
        double t = 0;
        for (int d = 1; d < head.length; d++) {
            t += s[head[d]][d];
        }
        return t;
    }

    private static boolean projective(int[] head) {
        for (int a = 1; a < head.length; a++) {
            int lo = Math.min(a, head[a]);
            int hi = Math.max(a, head[a]);
            for (int b = 1; b < head.length; b++) {
                int lo2 = Math.min(b, head[b]);
                int hi2 = Math.max(b, head[b]);
                if (lo < lo2 && lo2 < hi && hi < hi2 || lo2 < lo && lo < hi2 && hi2 < hi) {
                    return false;
                }
            }
        }
        return true;
    }

    /** Brute force over all head assignments: {best tree, best projective tree} scores. */
    private static double[] brute(double[][] s) {
        int n = s.length - 1;
        int[] head = new int[n + 1];
        double[] best = {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};
        int total = (int) Math.pow(n + 1, n);
        for (int code = 0; code < total; code++) {
            int c = code;
            for (int d = 1; d <= n; d++) {
                head[d] = c % (n + 1);
                c /= n + 1;
            }
            if (!BiaffineParser.isTree(Arrays.copyOfRange(head, 1, n + 1))) {
                continue;
            }
            double v = score(s, head);
            best[0] = Math.max(best[0], v);
            if (projective(head)) {
                best[1] = Math.max(best[1], v);
            }
        }
        return best;
    }

    @Test
    void decodersMatchBruteForce() {
        SplittableRandom r = new SplittableRandom(3);
        for (int trial = 0; trial < 200; trial++) {
            int n = 2 + r.nextInt(4);
            double[][] s = new double[n + 1][n + 1];
            for (int h = 0; h <= n; h++) {
                for (int d = 0; d <= n; d++) {
                    s[h][d] = d == 0 || h == d ? -1e30 : r.nextGaussian();
                }
            }
            double[] best = brute(s);
            int[] cle = BiaffineParser.chuLiuEdmonds(s);
            int[] eis = BiaffineParser.eisner(s);
            assertTrue(BiaffineParser.isTree(Arrays.copyOfRange(cle, 1, n + 1)), Arrays.toString(cle));
            assertTrue(BiaffineParser.isTree(Arrays.copyOfRange(eis, 1, n + 1)), Arrays.toString(eis));
            assertEquals(best[0], score(s, cle), 1e-9, "CLE trial " + trial);
            assertEquals(best[1], score(s, eis), 1e-9, "Eisner trial " + trial);
        }
    }

    @Test
    void insideOutsideMarginalsSumToExpectedSpanCount() {
        SplittableRandom r = new SplittableRandom(5);
        int n = 6;
        double[][] s = new double[n * (n + 1) / 2][3];
        for (double[] row : s) {
            for (int k = 0; k < 3; k++) {
                row[k] = r.nextGaussian();
            }
        }
        double[] z = new double[1];
        double[][] m = ChartParser.marginals(s, n, z);
        double total = 0;
        for (double[] row : m) {
            for (double v : row) {
                total += v;
            }
        }
        // every binary tree over n words has 2n - 1 spans
        assertEquals(2 * n - 1, total, 1e-6);
        int[] tree = ChartParser.cky(s, n);
        assertEquals(2 * n - 1, Arrays.stream(tree).filter(x -> x >= 0).count());
    }
}
