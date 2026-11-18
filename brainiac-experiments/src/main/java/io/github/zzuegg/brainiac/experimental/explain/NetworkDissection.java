package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Network dissection (Bau, Zhou, Khosla, Oliva and Torralba, CVPR 2017,
 * <a href="https://arxiv.org/abs/1704.05796">arXiv:1704.05796</a>) and compositional explanations
 * of units (Mu and Andreas, NeurIPS 2020, <a href="https://arxiv.org/abs/2006.14032">arXiv:2006.14032</a>).
 *
 * <p>Each convolutional unit's activation map is upsampled to the input grid ({@link #unitMaps})
 * and thresholded at its own top quantile over the whole dataset ({@code topShare}; Bau et al. use
 * 0.005 on large images, a small grid whose objects cover a few percent of the pixels needs a
 * larger share). The binary unit mask is scored against every concept's per-pixel segmentation
 * mask by the intersection over union accumulated over the dataset; a unit is a detector for its
 * best concept when that IoU exceeds a threshold (0.04 in the paper). The compositional search is
 * a beam search over formulas {@code F and C}, {@code F or C}, {@code F and not C} up to a given
 * number of concepts. {@link #randomDirections} gives the control: random linear combinations of
 * the layer's channels dissected the same way.
 *
 * <p>Masks are bitsets over (image, pixel), so a formula's IoU costs a few hundred word
 * operations.
 */
public final class NetworkDissection {

    /**
     * A unit's explanation.
     *
     * @param formula the concept or formula
     * @param iou its intersection over union with the unit's mask
     * @param length the number of concepts in the formula
     */
    public record Label(String formula, double iou, int length) {
    }

    /**
     * The dissection of one layer.
     *
     * @param single the best single concept per unit
     * @param compositional the best formula per unit
     */
    public record Dissection(Label[] single, Label[] compositional) {

        /**
         * @param threshold the detector IoU threshold
         * @return the number of distinct concepts some unit detects
         */
        public int uniqueDetectors(double threshold) {
            Set<String> s = new HashSet<>();
            for (Label l : single) {
                if (l.iou() > threshold) {
                    s.add(l.formula());
                }
            }
            return s.size();
        }

        /**
         * @param concept a concept name
         * @param threshold the detector IoU threshold
         * @return the number of units detecting it
         */
        public int detectors(String concept, double threshold) {
            int c = 0;
            for (Label l : single) {
                if (l.iou() > threshold && l.formula().equals(concept)) {
                    c++;
                }
            }
            return c;
        }

        /** @return the mean best single-concept IoU */
        public double meanIou() {
            return Arrays.stream(single).mapToDouble(Label::iou).average().orElse(0);
        }

        /** @return the mean best formula IoU */
        public double meanFormulaIou() {
            return Arrays.stream(compositional).mapToDouble(Label::iou).average().orElse(0);
        }

        /**
         * @param gain the relative IoU gain that counts
         * @return the share of units whose formula beats their single concept by the gain
         */
        public double formulaShare(double gain) {
            int c = 0;
            for (int u = 0; u < single.length; u++) {
                if (compositional[u].length() > 1 && compositional[u].iou() >= (1 + gain) * single[u].iou()) {
                    c++;
                }
            }
            return c / (double) single.length;
        }
    }

    private record Formula(String text, long[] bits, int length, double iou) {
    }

    private NetworkDissection() {
    }

    /**
     * Upsamples a layer's channel maps to the input grid by nearest neighbour, centred (valid
     * convolutions shrink the map evenly at every border); uncovered border pixels take the
     * nearest covered cell.
     *
     * @param activations {@code [n, channels, h, w]}
     * @param side the input side
     * @param stride input pixels per map cell
     * @return {@code [channels][n][side * side]}
     */
    public static float[][][] unitMaps(Tensor activations, int side, int stride) {
        int n = activations.shape().dim(0);
        int c = activations.shape().dim(1);
        int h = activations.shape().dim(2);
        int w = activations.shape().dim(3);
        float[] v = activations.toArray();
        float[][][] out = new float[c][n][side * side];
        int oy = (side - stride * h) / 2;
        int ox = (side - stride * w) / 2;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < c; k++) {
                int base = (i * c + k) * h * w;
                for (int y = 0; y < side; y++) {
                    int cy = Math.max(0, Math.min(h - 1, Math.floorDiv(y - oy, stride)));
                    for (int x = 0; x < side; x++) {
                        int cx = Math.max(0, Math.min(w - 1, Math.floorDiv(x - ox, stride)));
                        out[k][i][y * side + x] = v[base + cy * w + cx];
                    }
                }
            }
        }
        return out;
    }

    /**
     * Maps of random directions in the layer: each a Gaussian combination of its channels.
     *
     * @param activations {@code [n, channels, h, w]}
     * @param count the number of directions
     * @param side the input side
     * @param stride input pixels per map cell
     * @param seed the seed
     * @return {@code [count][n][side * side]}
     */
    public static float[][][] randomDirections(Tensor activations, int count, int side, int stride, long seed) {
        float[][][] units = unitMaps(activations, side, stride);
        int c = units.length;
        int n = units[0].length;
        int area = side * side;
        SplittableRandom r = new SplittableRandom(seed);
        float[][][] out = new float[count][n][area];
        for (int d = 0; d < count; d++) {
            double[] w = new double[c];
            for (int k = 0; k < c; k++) {
                w[k] = r.nextGaussian();
            }
            for (int i = 0; i < n; i++) {
                for (int p = 0; p < area; p++) {
                    double s = 0;
                    for (int k = 0; k < c; k++) {
                        s += w[k] * units[k][i][p];
                    }
                    out[d][i][p] = (float) s;
                }
            }
        }
        return out;
    }

    /**
     * Dissects a layer.
     *
     * @param maps {@code [units][n][area]} upsampled maps
     * @param masks {@code [n][concepts][area]} concept masks
     * @param names the concept names
     * @param topShare the share of each unit's activations kept as its mask
     * @param maxLength the most concepts per formula (1 skips the compositional search)
     * @param beam the beam width
     * @return the labels
     */
    public static Dissection dissect(float[][][] maps, boolean[][][] masks, String[] names, double topShare,
            int maxLength, int beam) {
        long[][] concepts = conceptBits(masks);
        Label[] single = new Label[maps.length];
        Label[] comp = new Label[maps.length];
        for (int u = 0; u < maps.length; u++) {
            long[] unit = threshold(maps[u], topShare);
            List<Formula> atoms = new ArrayList<>();
            for (int k = 0; k < concepts.length; k++) {
                atoms.add(new Formula(names[k], concepts[k], 1, iou(unit, concepts[k])));
            }
            atoms.sort(Comparator.comparingDouble(Formula::iou).reversed());
            Formula best = atoms.get(0);
            single[u] = new Label(best.text(), best.iou(), 1);
            List<Formula> frontier = new ArrayList<>(atoms.subList(0, Math.min(beam, atoms.size())));
            Formula top = best;
            for (int len = 2; len <= maxLength; len++) {
                List<Formula> next = new ArrayList<>(frontier);
                for (Formula f : frontier) {
                    for (int k = 0; k < concepts.length; k++) {
                        for (int op = 0; op < 3; op++) {
                            long[] b = combine(f.bits(), concepts[k], op);
                            String text = "(" + f.text() + (op == 0 ? " and " : op == 1 ? " or " : " and not ")
                                    + names[k] + ")";
                            next.add(new Formula(text, b, f.length() + 1, iou(unit, b)));
                        }
                    }
                }
                next.sort(Comparator.comparingDouble(Formula::iou).reversed());
                frontier = new ArrayList<>(next.subList(0, Math.min(beam, next.size())));
                if (frontier.get(0).iou() > top.iou()) {
                    top = frontier.get(0);
                }
            }
            comp[u] = new Label(top.text(), top.iou(), top.length());
        }
        return new Dissection(single, comp);
    }

    private static long[] combine(long[] a, long[] b, int op) {
        long[] out = new long[a.length];
        for (int w = 0; w < a.length; w++) {
            out[w] = switch (op) {
                case 0 -> a[w] & b[w];
                case 1 -> a[w] | b[w];
                default -> a[w] & ~b[w];
            };
        }
        return out;
    }

    private static long[][] conceptBits(boolean[][][] masks) {
        int n = masks.length;
        int k = masks[0].length;
        int area = masks[0][0].length;
        int words = (n * area + 63) / 64;
        long[][] out = new long[k][words];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < k; c++) {
                for (int p = 0; p < area; p++) {
                    if (masks[i][c][p]) {
                        int bit = i * area + p;
                        out[c][bit >> 6] |= 1L << (bit & 63);
                    }
                }
            }
        }
        return out;
    }

    /**
     * The unit mask: its activations above the {@code 1 - topShare} quantile over the dataset.
     *
     * @param map {@code [n][area]}
     * @param topShare the share kept
     * @return the bitset over (image, pixel)
     */
    static long[] threshold(float[][] map, double topShare) {
        int n = map.length;
        int area = map[0].length;
        float[] all = new float[n * area];
        for (int i = 0; i < n; i++) {
            System.arraycopy(map[i], 0, all, i * area, area);
        }
        float[] sorted = all.clone();
        Arrays.sort(sorted);
        int cut = Math.max(0, Math.min(sorted.length - 1, (int) Math.floor((1 - topShare) * sorted.length)));
        float t = sorted[cut];
        long[] out = new long[(all.length + 63) / 64];
        for (int b = 0; b < all.length; b++) {
            if (all[b] > t || (all[b] == t && t > 0)) {
                out[b >> 6] |= 1L << (b & 63);
            }
        }
        return out;
    }

    private static double iou(long[] a, long[] b) {
        long inter = 0;
        long union = 0;
        for (int w = 0; w < a.length; w++) {
            inter += Long.bitCount(a[w] & b[w]);
            union += Long.bitCount(a[w] | b[w]);
        }
        return union == 0 ? 0 : inter / (double) union;
    }
}
