package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.AnimatedGifWriter;
import io.github.zzuegg.brainiac.experimental.nn.LayerHooks;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Films a {@link Brain} while it trains (X31): add it as a listener and every few epochs it
 * draws a frame, then writes the frames as an animated GIF. Each frame shows
 *
 * <ul>
 *   <li>one panel per region, a grid with a cell per neuron coloured by its mean absolute
 *       activity on a fixed probe batch (a heat ramp; dormant neurons are dark grey), so you
 *       see regions light up, specialise and recruit new neurons;</li>
 *   <li>one panel per projection, its connection mask (source rows, target columns), with
 *       connections added since the last frame in green and removed ones in red, so
 *       pruning and regrowth by {@code Plasticity} are visible;</li>
 *   <li>the training-loss curve so far (log scale).</li>
 * </ul>
 *
 * <p>The activity is the brain's own per-region output on the probe
 * ({@link Brain#forwardRegions}) and dormancy its per-neuron state
 * ({@link Brain#activeNeurons(String)}).
 *
 * <p>Visualising network activity and structure over training follows the spirit of
 * Yosinski et al. 2015, "Understanding Neural Networks Through Deep Visualization",
 * <a href="https://arxiv.org/abs/1506.06579">arXiv:1506.06579</a>; the rewiring shown is
 * that of Evci et al. 2020, "Rigging the Lottery", <a href="https://arxiv.org/abs/1911.11134">arXiv:1911.11134</a>.
 */
public final class BrainMovie implements TrainingListener {

    private static final int BACKGROUND = 0;
    private static final int DORMANT = 1;
    private static final int KEPT = 2;
    private static final int ADDED = 3;
    private static final int REMOVED = 4;
    private static final int CURVE = 5;
    private static final int PANEL = 6;
    private static final int HEAT = 64;
    private static final int HEAT_LEVELS = 64;
    private static final int MARGIN = 6;
    private static final int CELL = 5;
    private static final int CURVE_HEIGHT = 40;

    private final Brain brain;
    private final Model model;
    private final int layerIndex;
    private final float[][] probe;
    private final Brain.Spec spec;
    private final int[] gridCols;
    private final int[] maskScale;
    private final int[][] regionAt;
    private final int[][] maskAt;
    private final int width;
    private final int height;
    private final List<byte[]> frames = new ArrayList<>();
    private final List<Double> losses = new ArrayList<>();
    private boolean[][] previousMasks;
    private double[][] lastActivity;
    private int every = 1;

    private BrainMovie(Brain brain, Model model, int layerIndex, float[][] probe) {
        this.brain = brain;
        this.model = model;
        this.layerIndex = layerIndex;
        this.probe = probe.clone();
        this.spec = brain.spec();
        int r = spec.regions().size();
        int p = spec.projections().size();
        gridCols = new int[r];
        regionAt = new int[r][];
        int x = MARGIN;
        int rowHeight = 0;
        for (int i = 0; i < r; i++) {
            int cap = spec.regions().get(i).capacity();
            gridCols[i] = (int) Math.ceil(Math.sqrt(cap));
            int rows = (cap + gridCols[i] - 1) / gridCols[i];
            regionAt[i] = new int[] {x, MARGIN, gridCols[i] * CELL + 2, rows * CELL + 2};
            x += regionAt[i][2] + MARGIN;
            rowHeight = Math.max(rowHeight, regionAt[i][3]);
        }
        int w = x;
        int y = MARGIN + rowHeight + MARGIN;
        x = MARGIN;
        maskScale = new int[p];
        maskAt = new int[p][];
        int maskHeight = 0;
        for (int j = 0; j < p; j++) {
            int cols = capacity(spec.projections().get(j).to());
            int rows = sourceSize(j);
            maskScale[j] = Math.max(1, Math.min(3, 96 / Math.max(rows, cols)));
            maskAt[j] = new int[] {x, y, cols * maskScale[j] + 2, rows * maskScale[j] + 2, rows, cols};
            x += maskAt[j][2] + MARGIN;
            maskHeight = Math.max(maskHeight, maskAt[j][3]);
        }
        this.width = Math.max(Math.max(w, x), 120);
        this.height = y + maskHeight + MARGIN + CURVE_HEIGHT + MARGIN;
    }

    /**
     * Films a brain that is fed directly.
     *
     * @param brain the brain
     * @param probe a fixed batch of brain inputs whose activity is drawn
     * @return the movie
     */
    public static BrainMovie of(Brain brain, float[][] probe) {
        return new BrainMovie(Objects.requireNonNull(brain), null, -1, probe);
    }

    /**
     * Films the first brain inside a sequential model.
     *
     * @param model the model
     * @param probe a fixed batch of model inputs; the layers before the brain are run on it
     *     for every frame
     * @return the movie
     */
    public static BrainMovie of(Model model, float[][] probe) {
        List<Layer> layers = model.layers();
        for (int i = 0; i < layers.size(); i++) {
            if (layers.get(i) instanceof Brain b) {
                return new BrainMovie(b, model, i, probe);
            }
        }
        throw new IllegalArgumentException("the model has no Brain layer");
    }

    /**
     * Draws a frame every {@code n} epochs.
     *
     * @param n the interval
     * @return this
     */
    public BrainMovie every(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("every must be positive");
        }
        this.every = n;
        return this;
    }

    @Override
    public void onEpoch(EpochResult result) {
        losses.add((double) result.loss());
        if (result.epoch() % every == 0) {
            capture();
        }
    }

    /** Draws a frame of the brain's current state (also usable before training starts). */
    public void capture() {
        byte[] px = new byte[width * height];
        lastActivity = activity();
        double max = 1e-9;
        for (double[] a : lastActivity) {
            for (double v : a) {
                max = Math.max(max, v);
            }
        }
        boolean[][] masks = new boolean[spec.projections().size()][];
        for (int j = 0; j < masks.length; j++) {
            masks[j] = brain.connectionMask(j);
        }
        boolean[][] active = activeNeurons();
        for (int i = 0; i < regionAt.length; i++) {
            int[] at = regionAt[i];
            fill(px, at[0], at[1], at[2], at[3], PANEL);
            for (int k = 0; k < lastActivity[i].length; k++) {
                int colour = active[i][k]
                        ? HEAT + (int) Math.min(HEAT_LEVELS - 1, lastActivity[i][k] / max * HEAT_LEVELS) : DORMANT;
                fill(px, at[0] + 1 + k % gridCols[i] * CELL, at[1] + 1 + k / gridCols[i] * CELL, CELL - 1, CELL - 1,
                        colour);
            }
        }
        for (int j = 0; j < maskAt.length; j++) {
            int[] at = maskAt[j];
            int s = maskScale[j];
            fill(px, at[0], at[1], at[2], at[3], PANEL);
            for (int a = 0; a < at[4]; a++) {
                for (int b = 0; b < at[5]; b++) {
                    int idx = a * at[5] + b;
                    boolean now = masks[j][idx];
                    boolean before = previousMasks == null ? now : previousMasks[j][idx];
                    int colour = now && before ? KEPT : now ? ADDED : before ? REMOVED : BACKGROUND;
                    if (colour != BACKGROUND) {
                        fill(px, at[0] + 1 + b * s, at[1] + 1 + a * s, s, s, colour);
                    }
                }
            }
        }
        drawCurve(px);
        previousMasks = masks;
        frames.add(px);
    }

    private void drawCurve(byte[] px) {
        int top = height - MARGIN - CURVE_HEIGHT;
        fill(px, MARGIN, top, width - 2 * MARGIN, CURVE_HEIGHT, PANEL);
        if (losses.isEmpty()) {
            return;
        }
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double l : losses) {
            double v = Math.log(Math.max(l, 1e-12));
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        int plotW = width - 2 * MARGIN - 2;
        int prevY = -1;
        for (int x = 0; x < plotW; x++) {
            double pos = losses.size() == 1 ? 0 : x * (losses.size() - 1) / (double) (plotW - 1);
            int i = Math.min((int) pos, losses.size() - 1);
            int j = Math.min(i + 1, losses.size() - 1);
            double v = (1 - (pos - i)) * Math.log(Math.max(losses.get(i), 1e-12))
                    + (pos - i) * Math.log(Math.max(losses.get(j), 1e-12));
            double t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
            int y = top + 1 + (int) Math.round((1 - t) * (CURVE_HEIGHT - 3));
            int from = prevY < 0 ? y : Math.min(prevY, y);
            int to = prevY < 0 ? y : Math.max(prevY, y);
            for (int yy = from; yy <= to; yy++) {
                px[yy * width + MARGIN + 1 + x] = CURVE;
            }
            prevY = y;
        }
    }

    private void fill(byte[] px, int x, int y, int w, int h, int colour) {
        for (int yy = y; yy < y + h && yy < height; yy++) {
            for (int xx = x; xx < x + w && xx < width; xx++) {
                px[yy * width + xx] = (byte) colour;
            }
        }
    }

    /**
     * The mean absolute activity of every neuron on the probe batch, from the brain's own
     * per-region outputs ({@link Brain#forwardRegions}).
     *
     * @return per region, one value per neuron (capacity)
     */
    public double[][] activity() {
        float[][] in = brainInput();
        int n = in.length;
        Map<String, Var> regions = brain.forwardRegions(Var.of(Tensor.of(in)), false);
        double[][] out = new double[regions.size()][];
        int i = 0;
        for (Var region : regions.values()) {
            float[] a = region.value().to(DType.FLOAT32).toArray();
            int cap = a.length / n;
            out[i] = new double[cap];
            for (int s = 0; s < n; s++) {
                for (int k = 0; k < cap; k++) {
                    out[i][k] += Math.abs(a[s * cap + k]) / n;
                }
            }
            i++;
        }
        return out;
    }

    private float[][] brainInput() {
        if (model == null || layerIndex == 0) {
            return probe;
        }
        List<Var> trace = LayerHooks.trace(model, Var.of(Tensor.of(probe)));
        Tensor h = trace.get(layerIndex - 1).value();
        float[] a = h.to(DType.FLOAT32).toArray();
        int f = a.length / probe.length;
        float[][] out = new float[probe.length][f];
        for (int s = 0; s < probe.length; s++) {
            System.arraycopy(a, s * f, out[s], 0, f);
        }
        return out;
    }

    private boolean[][] activeNeurons() {
        boolean[][] active = new boolean[spec.regions().size()][];
        for (int i = 0; i < active.length; i++) {
            active[i] = brain.activeNeurons(spec.regions().get(i).name());
        }
        return active;
    }

    private int regionIndex(String name) {
        for (int i = 0; i < spec.regions().size(); i++) {
            if (spec.regions().get(i).name().equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("no region " + name);
    }

    private int capacity(String region) {
        return spec.regions().get(regionIndex(region)).capacity();
    }

    private int sourceSize(int projection) {
        Brain.ProjectionSpec ps = spec.projections().get(projection);
        if (!ps.from().equals(Brain.INPUT)) {
            return capacity(ps.from());
        }
        return brain.weightValues(projection).length / capacity(ps.to());
    }

    /** @return the number of frames drawn */
    public int frames() {
        return frames.size();
    }

    /** @return the frame width in pixels */
    public int width() {
        return width;
    }

    /** @return the frame height in pixels */
    public int height() {
        return height;
    }

    /**
     * A copy of one frame's palette indices, row by row.
     *
     * @param index the frame
     * @return the pixels
     */
    public byte[] frame(int index) {
        return frames.get(index).clone();
    }

    /** @return the colours of the frames' palette as {@code 0xRRGGBB} */
    public static int[] palette() {
        int[] p = new int[HEAT + HEAT_LEVELS];
        p[BACKGROUND] = 0x101418;
        p[DORMANT] = 0x2A3038;
        p[KEPT] = 0x8A94A0;
        p[ADDED] = 0x3CCB5A;
        p[REMOVED] = 0xE5484D;
        p[CURVE] = 0xF2F2F2;
        p[PANEL] = 0x1C2229;
        // a black - purple - red - orange - yellow heat ramp
        int[][] stops = {{0, 0, 4}, {87, 16, 110}, {188, 55, 84}, {249, 142, 9}, {252, 255, 164}};
        for (int i = 0; i < HEAT_LEVELS; i++) {
            double t = i / (double) (HEAT_LEVELS - 1) * (stops.length - 1);
            int a = Math.min((int) t, stops.length - 2);
            double f = t - a;
            int rgb = 0;
            for (int c = 0; c < 3; c++) {
                rgb = rgb << 8 | (int) Math.round(stops[a][c] * (1 - f) + stops[a + 1][c] * f);
            }
            p[HEAT + i] = rgb;
        }
        return p;
    }

    /**
     * Encodes the frames as a looping GIF.
     *
     * @param delayCentiseconds how long each frame shows
     * @return the GIF's bytes
     */
    public byte[] toGif(int delayCentiseconds) {
        AnimatedGifWriter gif = AnimatedGifWriter.of(width, height, palette());
        for (byte[] f : frames) {
            gif.frame(f, delayCentiseconds);
        }
        return gif.toBytes();
    }

    /**
     * Writes the frames as a looping GIF file.
     *
     * @param file the file
     * @param delayCentiseconds how long each frame shows
     * @throws IOException if writing fails
     */
    public void write(Path file, int delayCentiseconds) throws IOException {
        java.nio.file.Files.write(file, toGif(delayCentiseconds));
    }
}
