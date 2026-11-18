package io.github.zzuegg.brainiac.experimental.privacy;

import java.math.BigInteger;
import java.util.SplittableRandom;

/**
 * Secure aggregation by pairwise masking (Bonawitz et al., CCS 2017, "Practical Secure
 * Aggregation for Privacy-Preserving Machine Learning", <a href="https://eprint.iacr.org/2017/281">ePrint 2017/281</a>),
 * simulated in-process. The server learns the sum of the clients' vectors and nothing else:
 *
 * <ol>
 *   <li>Every client {@code u} has a Diffie-Hellman key pair {@code (sk_u, g^sk_u mod p)} over
 *       the Mersenne prime {@code p = 2^61 - 1}; every pair agrees on the seed
 *       {@code s_uv = (g^sk_v)^sk_u}, which either of them can compute.</li>
 *   <li>Every client draws a self-mask seed {@code b_u} and Shamir-shares both {@code sk_u}
 *       and {@code b_u} among all clients with threshold {@code t}.</li>
 *   <li>Inputs are encoded in fixed point in the ring {@code Z_(2^32)} and sent masked:
 *       {@code y_u = x_u + PRG(b_u) + Σ_(v>u) PRG(s_uv) - Σ_(v<u) PRG(s_uv)}. Pair masks cancel
 *       in the sum.</li>
 *   <li>Clients may drop out after the setup. For each survivor the server collects {@code t}
 *       shares of {@code b_u} and removes its self-mask; for each dropped client it collects
 *       {@code t} shares of {@code sk_v}, recomputes the pair masks between {@code v} and the
 *       survivors and removes them. (An honest client never reveals both secrets of one
 *       client, so no single input is ever unmasked.)</li>
 * </ol>
 *
 * <p>The result is the exact fixed-point sum of the survivors' inputs, whatever the dropout
 * rate, as long as at least {@code t} clients survive.
 */
public final class SecureAggregation {

    /** The Mersenne prime {@code 2^61 - 1}, the field of the key agreement and secret sharing. */
    static final BigInteger PRIME = BigInteger.ONE.shiftLeft(61).subtract(BigInteger.ONE);
    private static final BigInteger GENERATOR = BigInteger.valueOf(37);

    /**
     * The outcome of one aggregation.
     *
     * @param sum the decoded sum of the survivors' inputs
     * @param survivors how many clients sent their masked input
     * @param recoveredSecrets secrets the server reconstructed from shares
     * @param maskExpansions mask vectors the clients and server generated
     * @param nanos the wall time of the whole protocol
     */
    public record Result(float[] sum, int survivors, int recoveredSecrets, long maskExpansions, long nanos) {
    }

    private final int clients;
    private final int threshold;
    private final double scale;
    private final SplittableRandom random;

    private SecureAggregation(int clients, int threshold, double scale, long seed) {
        if (threshold < 1 || threshold > clients) {
            throw new IllegalArgumentException("threshold must be in [1, clients]: " + threshold);
        }
        this.clients = clients;
        this.threshold = threshold;
        this.scale = scale;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates the protocol.
     *
     * @param clients the number of clients
     * @param threshold the fewest survivors that can recover the sum
     * @param scale the fixed-point scale; {@code |sum| * scale} must stay below {@code 2^31}
     * @param seed the seed of all clients' randomness
     * @return the protocol
     */
    public static SecureAggregation of(int clients, int threshold, double scale, long seed) {
        return new SecureAggregation(clients, threshold, scale, seed);
    }

    /**
     * Runs one aggregation.
     *
     * @param inputs every client's vector, same length
     * @param dropped which clients drop out after the setup (their input is not counted)
     * @return the sum of the survivors' inputs, as the server decodes it
     */
    public Result aggregate(float[][] inputs, boolean[] dropped) {
        long start = System.nanoTime();
        int n = clients;
        int d = inputs[0].length;
        // setup: key pairs, self-mask seeds and their shares
        BigInteger[] sk = new BigInteger[n];
        BigInteger[] pk = new BigInteger[n];
        long[] selfSeed = new long[n];
        BigInteger[][] skShares = new BigInteger[n][];
        BigInteger[][] bShares = new BigInteger[n][];
        for (int u = 0; u < n; u++) {
            sk[u] = BigInteger.valueOf(1 + random.nextLong(PRIME.longValue() - 2));
            pk[u] = GENERATOR.modPow(sk[u], PRIME);
            selfSeed[u] = random.nextLong(PRIME.longValue());
            skShares[u] = share(sk[u], n, threshold, random);
            bShares[u] = share(BigInteger.valueOf(selfSeed[u]), n, threshold, random);
        }
        long expansions = 0;
        // masked inputs from the survivors
        int[] total = new int[d];
        int survivors = 0;
        int[] mask = new int[d];
        for (int u = 0; u < n; u++) {
            if (dropped[u]) {
                continue;
            }
            survivors++;
            int[] y = encode(inputs[u]);
            expand(selfSeed[u], mask);
            expansions++;
            addInto(y, mask, 1);
            for (int v = 0; v < n; v++) {
                if (v != u) {
                    expand(pairSeed(sk[u], pk[v]), mask);
                    expansions++;
                    addInto(y, mask, v > u ? 1 : -1);
                }
            }
            addInto(total, y, 1);
        }
        if (survivors < threshold) {
            throw new IllegalStateException(survivors + " survivors cannot reach the threshold " + threshold);
        }
        // unmasking
        int[] holders = new int[threshold];
        for (int u = 0, k = 0; u < n && k < threshold; u++) {
            if (!dropped[u]) {
                holders[k++] = u;
            }
        }
        int recovered = 0;
        for (int u = 0; u < n; u++) {
            if (!dropped[u]) {
                long b = reconstruct(bShares[u], holders).longValue();
                recovered++;
                expand(b, mask);
                expansions++;
                addInto(total, mask, -1);
            } else {
                BigInteger key = reconstruct(skShares[u], holders);
                recovered++;
                for (int v = 0; v < n; v++) {
                    if (!dropped[v]) {
                        // survivor v added +PRG(s_vu) if u > v, else -PRG(s_vu)
                        expand(pairSeed(key, pk[v]), mask);
                        expansions++;
                        addInto(total, mask, u > v ? -1 : 1);
                    }
                }
            }
        }
        float[] sum = new float[d];
        for (int i = 0; i < d; i++) {
            sum[i] = (float) (total[i] / scale);
        }
        return new Result(sum, survivors, recovered, expansions, System.nanoTime() - start);
    }

    private int[] encode(float[] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (int) Math.round(x[i] * scale);
        }
        return out;
    }

    private static long pairSeed(BigInteger mySecret, BigInteger otherPublic) {
        return otherPublic.modPow(mySecret, PRIME).longValue();
    }

    /** Fills a mask vector from a seed: the protocol's pseudo-random generator. */
    private static void expand(long seed, int[] out) {
        SplittableRandom prg = new SplittableRandom(seed);
        for (int i = 0; i < out.length; i++) {
            out[i] = prg.nextInt();
        }
    }

    private static void addInto(int[] acc, int[] v, int sign) {
        for (int i = 0; i < acc.length; i++) {
            acc[i] += sign * v[i];   // wraps around: arithmetic in Z_(2^32)
        }
    }

    /**
     * Shamir's {@code t}-of-{@code n} sharing over {@code GF(2^61 - 1)}: shares are a random
     * polynomial of degree {@code t - 1} with the secret as constant term, evaluated at
     * {@code 1..n}.
     *
     * @param secret the secret
     * @param n the number of shares
     * @param t the threshold
     * @param random the random source
     * @return the shares, share {@code i} at {@code x = i + 1}
     */
    static BigInteger[] share(BigInteger secret, int n, int t, SplittableRandom random) {
        BigInteger[] coeff = new BigInteger[t];
        coeff[0] = secret.mod(PRIME);
        for (int i = 1; i < t; i++) {
            coeff[i] = BigInteger.valueOf(random.nextLong(PRIME.longValue()));
        }
        BigInteger[] out = new BigInteger[n];
        for (int x = 1; x <= n; x++) {
            BigInteger bx = BigInteger.valueOf(x);
            BigInteger y = BigInteger.ZERO;
            for (int i = t - 1; i >= 0; i--) {
                y = y.multiply(bx).add(coeff[i]).mod(PRIME);
            }
            out[x - 1] = y;
        }
        return out;
    }

    /**
     * Lagrange interpolation at 0 from the shares of the given holders.
     *
     * @param shares all shares
     * @param holders the indices of {@code t} share holders
     * @return the secret
     */
    static BigInteger reconstruct(BigInteger[] shares, int[] holders) {
        BigInteger secret = BigInteger.ZERO;
        for (int a : holders) {
            BigInteger num = BigInteger.ONE;
            BigInteger den = BigInteger.ONE;
            BigInteger xa = BigInteger.valueOf(a + 1L);
            for (int b : holders) {
                if (b != a) {
                    BigInteger xb = BigInteger.valueOf(b + 1L);
                    num = num.multiply(xb.negate()).mod(PRIME);
                    den = den.multiply(xa.subtract(xb)).mod(PRIME);
                }
            }
            secret = secret.add(shares[a].multiply(num).multiply(den.modInverse(PRIME))).mod(PRIME);
        }
        return secret;
    }
}
