package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Short clips of moving shapes for action recognition. Eight actions in four time-reversed
 * pairs (classes {@code 2k} and {@code 2k + 1}), each pair rendered over the same range of
 * states so that no single frame tells the two apart:
 *
 * <ul>
 *   <li>slide right / slide left (a square moving one pixel per frame);
 *   <li>grow / shrink (a disc whose radius changes);
 *   <li>approach / recede (two blobs on a line moving together or apart);
 *   <li>rotate clockwise / counter-clockwise (a bar turning 22.5 degrees per frame).
 * </ul>
 *
 * <p>Every clip also holds a static distractor blob of similar brightness, random intensities
 * and pixel noise.
 */
public final class ShapeClips {

    /** The number of actions. */
    public static final int ACTIONS = 8;

    /** The action names. */
    public static final List<String> NAMES = List.of("slide right", "slide left", "grow", "shrink", "approach",
            "recede", "rotate cw", "rotate ccw");

    /**
     * A clip.
     *
     * @param frames {@code frames * side * side} values, frame-major
     * @param action the action id
     */
    public record Clip(float[] frames, int action) {
    }

    private final int side;
    private final int length;
    private final float noise;

    /**
     * Creates the generator.
     *
     * @param side the frame side, at least 16
     * @param length frames per clip, at least 4
     * @param noise the pixel noise
     */
    public ShapeClips(int side, int length, float noise) {
        if (side < 16 || length < 4) {
            throw new IllegalArgumentException("side >= 16 and length >= 4 required");
        }
        this.side = side;
        this.length = length;
        this.noise = noise;
    }

    /** @return the frame side */
    public int side() {
        return side;
    }

    /** @return frames per clip */
    public int length() {
        return length;
    }

    private void blob(float[] f, int o, double cx, double cy, double r, float v) {
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) {
                    f[o + y * side + x] = Math.max(f[o + y * side + x], v);
                }
            }
        }
    }

    private Clip clip(int action, SplittableRandom r) {
        int n = side * side;
        float[] f = new float[length * n];
        float v = (float) (0.7 + 0.3 * r.nextDouble());
        double dx = 1 + r.nextInt(side - 4);
        double dy = 1 + r.nextInt(side - 4);
        double travel = length - 1;
        double x0 = 2 + r.nextDouble() * (side - 4 - travel);
        double y0 = 3 + r.nextDouble() * (side - 6);
        double cx = side / 2.0 + r.nextDouble() * 4 - 2;
        double cy = side / 2.0 + r.nextDouble() * 4 - 2;
        double angle0 = r.nextDouble() * Math.PI;
        float distractor = (float) (0.6 + 0.3 * r.nextDouble());
        for (int t = 0; t < length; t++) {
            int o = t * n;
            double s = (action & 1) == 0 ? t / travel : 1 - t / travel;      // forward or reversed time
            switch (action / 2) {
                case 0 -> {
                    double x = x0 + s * travel;
                    for (int yy = (int) y0 - 1; yy <= (int) y0 + 1; yy++) {
                        for (int xx = (int) Math.round(x) - 1; xx <= (int) Math.round(x) + 1; xx++) {
                            if (xx >= 0 && yy >= 0 && xx < side && yy < side) {
                                f[o + yy * side + xx] = v;
                            }
                        }
                    }
                }
                case 1 -> blob(f, o, cx, cy, 1.5 + 3.5 * s, v);
                case 2 -> {
                    double half = 6 - 4.5 * s;
                    blob(f, o, cx - half, cy, 1.3, v);
                    blob(f, o, cx + half, cy, 1.3, v);
                }
                default -> {
                    double a = angle0 + (action == 6 ? 1 : -1) * t * Math.PI / 8;
                    for (double u = -3.5; u <= 3.5; u += 0.25) {
                        int xx = (int) Math.round(cx + u * Math.cos(a));
                        int yy = (int) Math.round(cy + u * Math.sin(a));
                        if (xx >= 0 && yy >= 0 && xx < side && yy < side) {
                            f[o + yy * side + xx] = v;
                        }
                    }
                }
            }
            blob(f, o, dx + 1, dy + 1, 1.2, distractor);
        }
        for (int i = 0; i < f.length; i++) {
            f[i] += (float) (noise * r.nextGaussian());
        }
        return new Clip(f, action);
    }

    /**
     * Generates clips with balanced actions.
     *
     * @param count how many
     * @param seed the seed
     * @return the clips
     */
    public List<Clip> generate(int count, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Clip> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.add(clip(i % ACTIONS, r));
        }
        return out;
    }
}
