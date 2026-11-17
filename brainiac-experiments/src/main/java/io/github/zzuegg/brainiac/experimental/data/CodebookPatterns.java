package io.github.zzuegg.brainiac.experimental.data;

/**
 * Interleaving patterns for language models over the parallel token streams of a residual
 * vector-quantized codec (one stream per codebook, {@code codes[k][t]} for codebook
 * {@code k} at frame {@code t}), as compared in MusicGen (Copet et al.,
 * <a href="https://arxiv.org/abs/2306.05284">2306.05284</a>):
 *
 * <ul>
 *   <li>{@link #flatten}: all codebooks of a frame one after another, {@code K T} steps; exact
 *       autoregression, but {@code K} times longer sequences.</li>
 *   <li>{@link #delay}: codebook {@code k} shifted right by {@code k} steps, then all
 *       codebooks predicted in parallel at each step. {@code T + K - 1} steps; codebook
 *       {@code k} of frame {@code t} is predicted after codebooks {@code 0..k-1} of the same
 *       frame are known, which keeps most of the coarse-to-fine dependency.</li>
 * </ul>
 */
public final class CodebookPatterns {
    private CodebookPatterns() {
    }

    /**
     * The delay pattern.
     *
     * @param codes {@code [K][T]} codes
     * @param pad the id for the empty corners
     * @return {@code [K][T + K - 1]}: row {@code k} is the codes of codebook {@code k} shifted right by {@code k}
     */
    public static int[][] delay(int[][] codes, int pad) {
        int k = codes.length;
        int t = codes[0].length;
        int[][] out = new int[k][t + k - 1];
        for (int q = 0; q < k; q++) {
            java.util.Arrays.fill(out[q], pad);
            System.arraycopy(codes[q], 0, out[q], q, t);
        }
        return out;
    }

    /**
     * Undoes {@link #delay}.
     *
     * @param delayed {@code [K][T + K - 1]}
     * @return {@code [K][T]}
     */
    public static int[][] undelay(int[][] delayed) {
        int k = delayed.length;
        int t = delayed[0].length - k + 1;
        int[][] out = new int[k][t];
        for (int q = 0; q < k; q++) {
            System.arraycopy(delayed[q], q, out[q], 0, t);
        }
        return out;
    }

    /**
     * Frame by frame flattening with codebook-specific ids, {@code k * size + code}.
     *
     * @param codes {@code [K][T]}
     * @param size the codebook size
     * @return {@code [T * K]}
     */
    public static int[] flatten(int[][] codes, int size) {
        int k = codes.length;
        int t = codes[0].length;
        int[] out = new int[t * k];
        for (int f = 0; f < t; f++) {
            for (int q = 0; q < k; q++) {
                out[f * k + q] = q * size + codes[q][f];
            }
        }
        return out;
    }

    /**
     * Undoes {@link #flatten}.
     *
     * @param flat {@code [T * K]}
     * @param codebooks {@code K}
     * @param size the codebook size
     * @return {@code [K][T]}
     */
    public static int[][] unflatten(int[] flat, int codebooks, int size) {
        int t = flat.length / codebooks;
        int[][] out = new int[codebooks][t];
        for (int f = 0; f < t; f++) {
            for (int q = 0; q < codebooks; q++) {
                out[q][f] = flat[f * codebooks + q] - q * size;
            }
        }
        return out;
    }
}
