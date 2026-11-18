package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Error-correcting codes and channels for decoding experiments: binary linear block codes
 * (Hamming(7,4), the narrow-sense BCH(31,16) and BCH(63,45) codes built from their minimal
 * polynomials over GF(2^m), and a random regular LDPC code), the rate-1/2 constraint-length-3
 * convolutional code with generators (7, 5) in octal, BPSK over AWGN and Rayleigh fading, and
 * real-valued MIMO channel matrices (i.i.d. Rayleigh or with Kronecker correlation).
 *
 * <p>Block codes keep both a generator matrix {@code G} ({@code k x n}) and a parity-check
 * matrix {@code H} ({@code (n-k) x n}); the cyclic codes use the cyclic {@code H} built from the
 * shifts of the reversed check polynomial, which is the matrix neural belief propagation papers
 * start from (Nachmani, Be'ery and Burshtein 2016).
 */
public final class ChannelCodes {

    private ChannelCodes() {
    }

    /**
     * A binary linear block code.
     *
     * @param name a short name
     * @param n the block length
     * @param k the dimension
     * @param g the generator matrix {@code k x n}, entries 0/1
     * @param h the parity-check matrix {@code m x n}, entries 0/1, {@code H G^T = 0}
     */
    public record LinearCode(String name, int n, int k, int[][] g, int[][] h) {

        /** @return the rate {@code k / n} */
        public double rate() {
            return k / (double) n;
        }

        /**
         * Encodes message bits.
         *
         * @param message {@code k} bits
         * @return the codeword, {@code n} bits
         */
        public int[] encode(int[] message) {
            int[] c = new int[n];
            for (int i = 0; i < k; i++) {
                if (message[i] != 0) {
                    for (int j = 0; j < n; j++) {
                        c[j] ^= g[i][j];
                    }
                }
            }
            return c;
        }

        /**
         * The syndrome {@code H c}.
         *
         * @param word a hard-decision word
         * @return the syndrome bits
         */
        public int[] syndrome(int[] word) {
            int[] s = new int[h.length];
            for (int r = 0; r < h.length; r++) {
                int acc = 0;
                for (int j = 0; j < n; j++) {
                    acc ^= h[r][j] & word[j];
                }
                s[r] = acc;
            }
            return s;
        }

        /**
         * Every codeword, for maximum-likelihood decoding by enumeration (only for small k).
         *
         * @return {@code 2^k} codewords
         */
        public int[][] codewords() {
            if (k > 20) {
                throw new IllegalStateException("too many codewords: 2^" + k);
            }
            int[][] out = new int[1 << k][];
            for (int m = 0; m < out.length; m++) {
                int[] msg = new int[k];
                for (int i = 0; i < k; i++) {
                    msg[i] = (m >> i) & 1;
                }
                out[m] = encode(msg);
            }
            return out;
        }
    }

    /** @return Hamming(7,4) as the cyclic code with generator {@code x^3 + x + 1} */
    public static LinearCode hamming74() {
        return cyclic("Hamming(7,4)", 7, new int[] {1, 1, 0, 1});
    }

    /** @return the narrow-sense BCH(31,16) code, which corrects 3 errors */
    public static LinearCode bch31x16() {
        return bch(5, 0b100101, 3);
    }

    /** @return the narrow-sense BCH(63,45) code, which corrects 3 errors */
    public static LinearCode bch63x45() {
        return bch(6, 0b1000011, 3);
    }

    /**
     * A narrow-sense primitive BCH code: the generator polynomial is the least common multiple
     * of the minimal polynomials of {@code alpha, alpha^3, ..., alpha^(2t-1)}.
     *
     * @param m the field degree ({@code n = 2^m - 1})
     * @param primitive the primitive polynomial of GF(2^m), bit i the coefficient of {@code x^i}
     * @param t the designed error-correcting capability
     * @return the code
     */
    public static LinearCode bch(int m, int primitive, int t) {
        int n = (1 << m) - 1;
        int[] exp = new int[2 * n];
        int[] log = new int[n + 1];
        int x = 1;
        for (int i = 0; i < n; i++) {
            exp[i] = x;
            exp[i + n] = x;
            log[x] = i;
            x <<= 1;
            if ((x & (1 << m)) != 0) {
                x ^= primitive;
            }
        }
        boolean[] used = new boolean[n];
        int[] gen = {1};
        for (int i = 1; i <= 2 * t - 1; i += 2) {
            if (used[i % n]) {
                continue;
            }
            // cyclotomic coset of i, and the minimal polynomial prod (x - alpha^j) over GF(2^m)
            List<Integer> coset = new ArrayList<>();
            int j = i % n;
            while (!used[j]) {
                used[j] = true;
                coset.add(j);
                j = (2 * j) % n;
            }
            int[] poly = {1};
            for (int e : coset) {
                int[] next = new int[poly.length + 1];
                for (int a = 0; a < poly.length; a++) {
                    next[a + 1] ^= poly[a];
                    next[a] ^= poly[a] == 0 ? 0 : exp[log[poly[a]] + e];
                }
                poly = next;
            }
            int[] binary = new int[poly.length];
            for (int a = 0; a < poly.length; a++) {
                if (poly[a] > 1) {
                    throw new IllegalStateException("minimal polynomial not binary");
                }
                binary[a] = poly[a];
            }
            gen = multiply(gen, binary);
        }
        return cyclic("BCH(" + n + "," + (n - gen.length + 1) + ")", n, gen);
    }

    private static int[] multiply(int[] a, int[] b) {
        int[] out = new int[a.length + b.length - 1];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                out[i + j] ^= a[i] & b[j];
            }
        }
        return out;
    }

    /**
     * A cyclic code from its generator polynomial.
     *
     * @param name the name
     * @param n the length
     * @param gen the generator polynomial, index i the coefficient of {@code x^i}
     * @return the code with the cyclic {@code G} and {@code H}
     */
    public static LinearCode cyclic(String name, int n, int[] gen) {
        int r = gen.length - 1;
        int k = n - r;
        int[][] g = new int[k][n];
        for (int i = 0; i < k; i++) {
            System.arraycopy(gen, 0, g[i], i, gen.length);
        }
        // h(x) = (x^n + 1) / g(x)
        int[] rem = new int[n + 1];
        rem[0] = 1;
        rem[n] = 1;
        int[] h = new int[k + 1];
        for (int d = n; d >= r; d--) {
            if (rem[d] != 0) {
                h[d - r] = 1;
                for (int a = 0; a <= r; a++) {
                    rem[d - r + a] ^= gen[a];
                }
            }
        }
        int[][] hm = new int[r][n];
        for (int i = 0; i < r; i++) {
            for (int a = 0; a <= k; a++) {
                hm[i][i + a] = h[k - a];
            }
        }
        return new LinearCode(name, n, k, g, hm);
    }

    /**
     * A random regular LDPC code by the configuration model (Gallager-style column weight
     * {@code dv}, row weight {@code dc}), rejecting double edges, with the generator from the
     * null space of {@code H}.
     *
     * @param n the block length (a multiple of {@code dc})
     * @param dv the column weight
     * @param dc the row weight
     * @param seed the seed
     * @return the code (its rate is at least {@code 1 - dv/dc})
     */
    public static LinearCode regularLdpc(int n, int dv, int dc, long seed) {
        int m = n * dv / dc;
        SplittableRandom rnd = new SplittableRandom(seed);
        for (int attempt = 0; attempt < 1000; attempt++) {
            int[] sockets = new int[n * dv];
            for (int i = 0; i < sockets.length; i++) {
                sockets[i] = i / dv;
            }
            for (int i = sockets.length - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1);
                int t = sockets[i];
                sockets[i] = sockets[j];
                sockets[j] = t;
            }
            int[][] h = new int[m][n];
            boolean ok = true;
            for (int i = 0; i < sockets.length && ok; i++) {
                int row = i / dc;
                if (h[row][sockets[i]] != 0) {
                    ok = false;
                }
                h[row][sockets[i]] = 1;
            }
            if (ok) {
                int[][] g = nullSpace(h, n);
                return new LinearCode("LDPC(" + n + "," + g.length + ")", n, g.length, g, h);
            }
        }
        throw new IllegalStateException("no simple LDPC graph found");
    }

    /**
     * A basis of the binary null space of a matrix.
     *
     * @param a the matrix, rows of length {@code n}
     * @param n the number of columns
     * @return basis vectors {@code x} with {@code a x = 0}
     */
    public static int[][] nullSpace(int[][] a, int n) {
        int[][] r = new int[a.length][];
        for (int i = 0; i < a.length; i++) {
            r[i] = a[i].clone();
        }
        int[] pivotCol = new int[r.length];
        int rank = 0;
        for (int c = 0; c < n && rank < r.length; c++) {
            int p = -1;
            for (int i = rank; i < r.length; i++) {
                if (r[i][c] != 0) {
                    p = i;
                    break;
                }
            }
            if (p < 0) {
                continue;
            }
            int[] t = r[p];
            r[p] = r[rank];
            r[rank] = t;
            for (int i = 0; i < r.length; i++) {
                if (i != rank && r[i][c] != 0) {
                    for (int j = 0; j < n; j++) {
                        r[i][j] ^= r[rank][j];
                    }
                }
            }
            pivotCol[rank++] = c;
        }
        boolean[] isPivot = new boolean[n];
        for (int i = 0; i < rank; i++) {
            isPivot[pivotCol[i]] = true;
        }
        List<int[]> basis = new ArrayList<>();
        for (int f = 0; f < n; f++) {
            if (isPivot[f]) {
                continue;
            }
            int[] v = new int[n];
            v[f] = 1;
            for (int i = 0; i < rank; i++) {
                v[pivotCol[i]] = r[i][f];
            }
            basis.add(v);
        }
        return basis.toArray(new int[0][]);
    }

    /**
     * The rate-1/2, constraint-length-3 convolutional encoder with generators (7, 5) octal,
     * terminated with two zero tail bits.
     *
     * @param bits the message
     * @return {@code 2 (len + 2)} coded bits, interleaved {@code (c0, c1)} per step
     */
    public static int[] convolutionalEncode(int[] bits) {
        int[] out = new int[2 * (bits.length + 2)];
        int s1 = 0;
        int s2 = 0;
        for (int t = 0; t < bits.length + 2; t++) {
            int u = t < bits.length ? bits[t] : 0;
            out[2 * t] = u ^ s1 ^ s2;
            out[2 * t + 1] = u ^ s2;
            s2 = s1;
            s1 = u;
        }
        return out;
    }

    /**
     * BPSK: bit 0 to {@code +1}, bit 1 to {@code -1}.
     *
     * @param bits the bits
     * @return the symbols
     */
    public static double[] bpsk(int[] bits) {
        double[] x = new double[bits.length];
        for (int i = 0; i < bits.length; i++) {
            x[i] = 1 - 2 * bits[i];
        }
        return x;
    }

    /**
     * The AWGN noise standard deviation for unit-energy BPSK at a given {@code Eb/N0}.
     *
     * @param ebN0Db {@code Eb/N0} in dB
     * @param rate the code rate
     * @return {@code sigma = sqrt(1 / (2 R Eb/N0))}
     */
    public static double sigma(double ebN0Db, double rate) {
        return Math.sqrt(1 / (2 * rate * Math.pow(10, ebN0Db / 10)));
    }

    /**
     * Adds white Gaussian noise.
     *
     * @param x the transmitted symbols
     * @param sigma the noise standard deviation
     * @param rnd the random source
     * @return the received samples
     */
    public static double[] awgn(double[] x, double sigma, SplittableRandom rnd) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] + sigma * gaussian(rnd);
        }
        return y;
    }

    /**
     * Flat Rayleigh fading per symbol followed by AWGN; the receiver knows the gains.
     *
     * @param x the symbols
     * @param sigma the noise standard deviation
     * @param gains filled with the Rayleigh amplitudes (mean square 1)
     * @param rnd the random source
     * @return the received samples
     */
    public static double[] rayleigh(double[] x, double sigma, double[] gains, SplittableRandom rnd) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double a = gaussian(rnd);
            double b = gaussian(rnd);
            gains[i] = Math.sqrt((a * a + b * b) / 2);
            y[i] = gains[i] * x[i] + sigma * gaussian(rnd);
        }
        return y;
    }

    /**
     * Channel log-likelihood ratios {@code log P(bit 0) / P(bit 1) = 2 y / sigma^2}.
     *
     * @param y received BPSK samples
     * @param sigma the noise standard deviation
     * @return the LLRs
     */
    public static double[] llr(double[] y, double sigma) {
        double[] l = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            l[i] = 2 * y[i] / (sigma * sigma);
        }
        return l;
    }

    /**
     * A real-valued Rayleigh MIMO channel {@code H} ({@code rx x tx}) with i.i.d. entries of
     * variance {@code 1 / rx}, optionally with Kronecker correlation {@code rho^|i-j|} on both
     * sides ({@code H = R_r^(1/2) H_w R_t^(1/2)}). A complex {@code N x N} channel with QPSK is
     * the real {@code 2N x 2N} channel with BPSK per real dimension.
     *
     * @param rx receive antennas
     * @param tx transmit antennas
     * @param rho the correlation coefficient, 0 for i.i.d.
     * @param rnd the random source
     * @return the matrix
     */
    public static double[][] mimoChannel(int rx, int tx, double rho, SplittableRandom rnd) {
        double[][] w = new double[rx][tx];
        for (double[] row : w) {
            for (int j = 0; j < tx; j++) {
                row[j] = gaussian(rnd) / Math.sqrt(rx);
            }
        }
        if (rho == 0) {
            return w;
        }
        double[][] lr = choleskyCorrelation(rx, rho);
        double[][] lt = choleskyCorrelation(tx, rho);
        double[][] out = new double[rx][tx];
        for (int i = 0; i < rx; i++) {
            for (int j = 0; j < tx; j++) {
                double s = 0;
                for (int a = 0; a < rx; a++) {
                    for (int b = 0; b < tx; b++) {
                        s += lr[i][a] * w[a][b] * lt[j][b];
                    }
                }
                out[i][j] = s;
            }
        }
        return out;
    }

    private static double[][] choleskyCorrelation(int n, double rho) {
        double[][] r = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                r[i][j] = Math.pow(rho, Math.abs(i - j));
            }
        }
        double[][] l = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double s = r[i][j];
                for (int a = 0; a < j; a++) {
                    s -= l[i][a] * l[j][a];
                }
                l[i][j] = i == j ? Math.sqrt(Math.max(s, 1e-12)) : s / l[j][j];
            }
        }
        return l;
    }

    /**
     * Random message bits.
     *
     * @param k the number of bits
     * @param rnd the random source
     * @return the bits
     */
    public static int[] randomBits(int k, SplittableRandom rnd) {
        int[] b = new int[k];
        for (int i = 0; i < k; i++) {
            b[i] = rnd.nextInt(2);
        }
        return b;
    }

    /**
     * The Hamming distance.
     *
     * @param a bits
     * @param b bits
     * @return the number of differing positions
     */
    public static int errors(int[] a, int[] b) {
        int e = 0;
        for (int i = 0; i < a.length; i++) {
            e += a[i] != b[i] ? 1 : 0;
        }
        return e;
    }

    /**
     * A standard normal draw.
     *
     * @param rnd the random source
     * @return the draw
     */
    public static double gaussian(SplittableRandom rnd) {
        double u = rnd.nextDouble();
        double v = rnd.nextDouble();
        return Math.sqrt(-2 * Math.log(1 - u)) * Math.cos(2 * Math.PI * v);
    }

    /**
     * The minimum Hamming distance of a small code, by enumeration.
     *
     * @param code the code
     * @return the minimum weight of a non-zero codeword
     */
    public static int minimumDistance(LinearCode code) {
        int best = Integer.MAX_VALUE;
        int[][] words = code.codewords();
        for (int i = 1; i < words.length; i++) {
            best = Math.min(best, Arrays.stream(words[i]).sum());
        }
        return best;
    }
}
