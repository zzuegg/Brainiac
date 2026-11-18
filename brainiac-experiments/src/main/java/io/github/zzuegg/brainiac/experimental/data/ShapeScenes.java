package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Synthetic multi-object scenes for detection, dense prediction, tracking and video toys on a
 * CPU. It extends the two-shape scenes of the {@code ShapeSegmentation} experiment (the six
 * {@link ShapeImages} glyphs on a noisy background) to everything the vision blocks need:
 *
 * <ul>
 *   <li>{@link #scene}: 1 to 5 glyphs at three scales (3, 6 and 9 pixels), with their boxes,
 *       classes, an instance mask and a semantic mask, laid out sparsely, crowded (overlaps
 *       allowed) or in touching same-class pairs;</li>
 *   <li>{@link #pose}: stick figures of 5 to 9 joints with random limb angles, joint
 *       coordinates and optional occluding bars;</li>
 *   <li>{@link #stereo}: a textured left/right pair of shapes at random depths with the exact
 *       disparity of every left pixel;</li>
 *   <li>{@link #video}: short videos of textured shapes moving with constant velocities and
 *       bouncing off the walls, with the exact optical flow between frames, boxes and track
 *       ids.</li>
 * </ul>
 *
 * <p>Boxes are {@code [x1, y1, x2, y2]} in pixel edges, so a glyph covering columns 4 to 6
 * has {@code x1 = 4, x2 = 7}. With {@code side = 12}, scale 1 and two shapes, {@link #scene}
 * gives the {@code ShapeSegmentation} setting.
 */
public final class ShapeScenes {

    /** The glyph classes, as in {@link ShapeImages}. */
    public static final int CLASSES = ShapeImages.CLASSES;
    /** The glyph scales: a glyph is 3 cells of {@code scale} pixels per side. */
    public static final int SCALES = 3;

    /** How the glyphs of a scene are placed. */
    public enum Layout {
        /** No two boxes overlap or touch. */
        SPARSE,
        /** Boxes may overlap; later glyphs are drawn over earlier ones. */
        CROWDED,
        /** Glyphs come in touching pairs of the same class and scale. */
        TOUCHING
    }

    private static final boolean[][] GLYPHS = {
        {false, false, false, true, true, true, false, false, false},
        {false, true, false, false, true, false, false, true, false},
        {true, true, true, true, false, true, true, true, true},
        {true, false, false, false, true, false, false, false, true},
        {false, false, true, false, true, false, true, false, false},
        {false, true, false, true, true, true, false, true, false}};

    private final int side;
    private final float noise;
    private final int maxScale;

    /**
     * Creates the generator with all three glyph scales.
     *
     * @param side the image side, at least 12
     * @param noise the standard deviation of the pixel noise, such as 0.1
     */
    public ShapeScenes(int side, float noise) {
        this(side, noise, SCALES);
    }

    /**
     * Creates the generator.
     *
     * @param side the image side, at least 12
     * @param noise the standard deviation of the pixel noise, such as 0.1
     * @param maxScale the largest glyph scale, 1 to 3
     */
    public ShapeScenes(int side, float noise, int maxScale) {
        if (side < 12) {
            throw new IllegalArgumentException("side must be at least 12, got " + side);
        }
        if (maxScale < 1 || maxScale > SCALES) {
            throw new IllegalArgumentException("maxScale must be 1 to " + SCALES + ", got " + maxScale);
        }
        this.side = side;
        this.noise = noise;
        this.maxScale = maxScale;
    }

    /** @return the image side */
    public int side() {
        return side;
    }

    /**
     * One scene of glyphs.
     *
     * @param side the image side
     * @param image {@code side * side} pixels, row-major
     * @param boxes {@code 4 k} box corners of the {@code k} glyphs
     * @param classes the class of each glyph
     * @param scales the scale of each glyph, 1 to 3
     * @param instances per pixel 0 for background or {@code 1 + glyph index}
     */
    public record Scene(int side, float[] image, float[] boxes, int[] classes, int[] scales, int[] instances) {

        /** @return the number of glyphs */
        public int count() {
            return classes.length;
        }

        /** @return per pixel 0 for background or {@code 1 + class} of the visible glyph */
        public int[] semantic() {
            int[] s = new int[instances.length];
            for (int i = 0; i < s.length; i++) {
                s[i] = instances[i] == 0 ? 0 : classes[instances[i] - 1] + 1;
            }
            return s;
        }

        /**
         * @param i a glyph
         * @return its box {@code [x1, y1, x2, y2]}
         */
        public float[] box(int i) {
            return new float[] {boxes[4 * i], boxes[4 * i + 1], boxes[4 * i + 2], boxes[4 * i + 3]};
        }

        /**
         * @param i a glyph
         * @return the full {@code 3 scale} square the glyph is drawn in, {@code [x1, y1, x2, y2]}
         */
        public float[] square(int i) {
            int[] t = tightBox(classes[i], scales[i], 0, 0);
            float x = boxes[4 * i] - t[0];
            float y = boxes[4 * i + 1] - t[1];
            return new float[] {x, y, x + 3 * scales[i], y + 3 * scales[i]};
        }

        /**
         * The tight box of a glyph of a class and scale drawn in a square.
         *
         * @param cls the class
         * @param scale the scale
         * @param x the square's left edge
         * @param y the square's top edge
         * @return {@code [x1, y1, x2, y2]}
         */
        public static float[] glyphBox(int cls, int scale, float x, float y) {
            int[] t = tightBox(cls, scale, 0, 0);
            return new float[] {x + t[0], y + t[1], x + t[2], y + t[3]};
        }
    }

    /**
     * Generates a scene.
     *
     * @param minShapes the fewest glyphs, at least 1
     * @param maxShapes the most glyphs
     * @param layout how the glyphs are placed
     * @param random the random source
     * @return the scene
     */
    public Scene scene(int minShapes, int maxShapes, Layout layout, SplittableRandom random) {
        int target = minShapes + random.nextInt(maxShapes - minShapes + 1);
        float[] image = new float[side * side];
        int[] inst = new int[side * side];
        List<int[]> placed = new ArrayList<>();
        int attempts = 0;
        while (placed.size() < target && attempts++ < 200) {
            int cls = random.nextInt(CLASSES);
            int scale = 1 + random.nextInt(maxScale);
            int extent = 3 * scale;
            int x = random.nextInt(side - extent + 1);
            int y = random.nextInt(side - extent + 1);
            if (layout == Layout.TOUCHING) {
                if (placed.size() + 2 > Math.max(2, target)) {
                    break;
                }
                int[] t = tightBox(cls, scale, x, y);
                boolean vertical = random.nextBoolean();
                int dx = vertical ? 0 : t[2] - t[0];
                int dy = vertical ? t[3] - t[1] : 0;
                int[] t2 = tightBox(cls, scale, x + dx, y + dy);
                if (t2[2] > side || t2[3] > side || !free(placed, t, 1) || !free(placed, t2, 1)) {
                    continue;
                }
                placed.add(new int[] {cls, scale, x, y});
                placed.add(new int[] {cls, scale, x + dx, y + dy});
                continue;
            }
            if (layout == Layout.SPARSE && !free(placed, tightBox(cls, scale, x, y), 1)) {
                continue;
            }
            placed.add(new int[] {cls, scale, x, y});
        }
        int k = placed.size();
        float[] boxes = new float[4 * k];
        int[] classes = new int[k];
        int[] scales = new int[k];
        for (int i = 0; i < k; i++) {
            int[] p = placed.get(i);
            classes[i] = p[0];
            scales[i] = p[1];
            int[] t = tightBox(p[0], p[1], p[2], p[3]);
            for (int j = 0; j < 4; j++) {
                boxes[4 * i + j] = t[j];
            }
            float bright = 0.7f + 0.6f * random.nextFloat();
            boolean[] g = GLYPHS[p[0]];
            for (int cy = 0; cy < 3; cy++) {
                for (int cx = 0; cx < 3; cx++) {
                    if (!g[cy * 3 + cx]) {
                        continue;
                    }
                    for (int yy = 0; yy < p[1]; yy++) {
                        for (int xx = 0; xx < p[1]; xx++) {
                            int px = p[2] + cx * p[1] + xx;
                            int py = p[3] + cy * p[1] + yy;
                            image[py * side + px] = bright;
                            inst[py * side + px] = i + 1;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < image.length; i++) {
            image[i] += (float) (noise * random.nextGaussian());
        }
        return new Scene(side, image, boxes, classes, scales, inst);
    }

    /**
     * Generates scenes.
     *
     * @param n how many
     * @param minShapes the fewest glyphs per scene
     * @param maxShapes the most glyphs per scene
     * @param layout how glyphs are placed
     * @param seed the seed
     * @return the scenes
     */
    public Scene[] scenes(int n, int minShapes, int maxShapes, Layout layout, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Scene[] out = new Scene[n];
        for (int i = 0; i < n; i++) {
            out[i] = scene(minShapes, maxShapes, layout, r);
        }
        return out;
    }

    /**
     * Stacks scene images.
     *
     * @param scenes the scenes, all of one side
     * @return {@code [n, 1, side, side]}
     */
    public static Tensor images(Scene[] scenes) {
        int s = scenes[0].side();
        float[] out = new float[scenes.length * s * s];
        for (int i = 0; i < scenes.length; i++) {
            System.arraycopy(scenes[i].image(), 0, out, i * s * s, s * s);
        }
        return Tensor.of(Shape.of(scenes.length, 1, s, s), out);
    }

    private static int[] tightBox(int cls, int scale, int x, int y) {
        boolean[] g = GLYPHS[cls];
        int x1 = 3;
        int y1 = 3;
        int x2 = 0;
        int y2 = 0;
        for (int cy = 0; cy < 3; cy++) {
            for (int cx = 0; cx < 3; cx++) {
                if (g[cy * 3 + cx]) {
                    x1 = Math.min(x1, cx);
                    y1 = Math.min(y1, cy);
                    x2 = Math.max(x2, cx + 1);
                    y2 = Math.max(y2, cy + 1);
                }
            }
        }
        return new int[] {x + x1 * scale, y + y1 * scale, x + x2 * scale, y + y2 * scale};
    }

    private boolean free(List<int[]> placed, int[] box, int gap) {
        for (int[] p : placed) {
            int[] o = tightBox(p[0], p[1], p[2], p[3]);
            if (box[0] < o[2] + gap && o[0] < box[2] + gap && box[1] < o[3] + gap && o[1] < box[3] + gap) {
                return false;
            }
        }
        return true;
    }

    // ------------------------------------------------------------------ stick figures

    /**
     * A stick figure.
     *
     * @param side the image side
     * @param image {@code side * side} pixels
     * @param joints {@code 2 J} joint coordinates {@code (x, y)} in pixels (pixel centres at
     *     {@code i + 0.5})
     * @param rare whether the figure was drawn from the rare (upside-down) pose mode
     */
    public record Pose(int side, float[] image, float[] joints, boolean rare) {

        /** @return the number of joints */
        public int jointCount() {
            return joints.length / 2;
        }
    }

    /**
     * Generates a stick figure: a chain of {@code joints} joints from a root near the top, each
     * limb pointing mostly downwards at a random angle (or, with probability {@code rareShare},
     * mostly upwards, the rare mode). Limbs are drawn dim, joints as bright dots whose
     * brightness encodes their index, so each joint is identifiable; {@code occluders} random
     * mid-grey bars are drawn on top.
     *
     * @param joints the joint count, 5 to 9
     * @param occluders the number of occluding bars
     * @param rareShare the probability of the rare pose mode
     * @param random the random source
     * @return the figure
     */
    public Pose pose(int joints, int occluders, double rareShare, SplittableRandom random) {
        boolean rare = random.nextDouble() < rareShare;
        float[] j = new float[2 * joints];
        float limb = side / (float) (joints + 1) * 1.1f;
        for (int attempt = 0; attempt < 100; attempt++) {
            j[0] = side * (0.3f + 0.4f * random.nextFloat());
            j[1] = rare ? side * (0.85f - 0.05f * random.nextFloat()) : side * (0.15f + 0.05f * random.nextFloat());
            boolean inside = true;
            for (int k = 1; k < joints && inside; k++) {
                double angle = Math.PI / 2 + (random.nextDouble() - 0.5) * 2.2;
                double len = limb * (0.8 + 0.4 * random.nextDouble());
                j[2 * k] = (float) (j[2 * k - 2] + len * Math.cos(angle));
                j[2 * k + 1] = (float) (j[2 * k - 1] + (rare ? -1 : 1) * len * Math.sin(angle));
                inside = j[2 * k] > 1 && j[2 * k] < side - 1 && j[2 * k + 1] > 1 && j[2 * k + 1] < side - 1;
            }
            if (inside) {
                break;
            }
        }
        float[] image = new float[side * side];
        for (int k = 1; k < joints; k++) {
            line(image, j[2 * k - 2], j[2 * k - 1], j[2 * k], j[2 * k + 1], 0.3f);
        }
        for (int k = 0; k < joints; k++) {
            float b = 0.5f + 0.5f * k / (joints - 1);
            int cx = (int) j[2 * k];
            int cy = (int) j[2 * k + 1];
            for (int dy = -1; dy <= 0; dy++) {
                for (int dx = -1; dx <= 0; dx++) {
                    int x = Math.min(side - 1, Math.max(0, cx + dx + (j[2 * k] - cx >= 0.5f ? 1 : 0)));
                    int y = Math.min(side - 1, Math.max(0, cy + dy + (j[2 * k + 1] - cy >= 0.5f ? 1 : 0)));
                    image[y * side + x] = b;
                }
            }
        }
        for (int o = 0; o < occluders; o++) {
            boolean horizontal = random.nextBoolean();
            int len = 3 + random.nextInt(side / 3);
            int a = random.nextInt(side);
            int b = random.nextInt(side - len);
            for (int t = 0; t < len; t++) {
                int x = horizontal ? b + t : a;
                int y = horizontal ? a : b + t;
                image[y * side + x] = 0.4f;
            }
        }
        for (int i = 0; i < image.length; i++) {
            image[i] += (float) (noise * random.nextGaussian());
        }
        return new Pose(side, image, j, rare);
    }

    private void line(float[] image, float x0, float y0, float x1, float y1, float value) {
        int steps = (int) Math.ceil(Math.hypot(x1 - x0, y1 - y0) * 2) + 1;
        for (int s = 0; s <= steps; s++) {
            float t = s / (float) steps;
            int x = (int) (x0 + t * (x1 - x0));
            int y = (int) (y0 + t * (y1 - y0));
            if (x >= 0 && x < side && y >= 0 && y < side) {
                image[y * side + x] = Math.max(image[y * side + x], value);
            }
        }
    }

    // ------------------------------------------------------------------ stereo pairs

    /**
     * A rectified stereo pair: a scene point at left column {@code x} appears at right column
     * {@code x - d}.
     *
     * @param side the image side
     * @param left the left image
     * @param right the right image
     * @param disparity the disparity of every left pixel
     */
    public record Stereo(int side, float[] left, float[] right, float[] disparity) {
    }

    /**
     * Generates a stereo pair: a random-texture background at disparity 0 and 1 to 3 textured
     * rectangles at integer disparities from 1 to {@code maxDisparity}, nearer ones drawn over
     * farther ones.
     *
     * @param maxDisparity the largest disparity
     * @param random the random source
     * @return the pair
     */
    public Stereo stereo(int maxDisparity, SplittableRandom random) {
        float[] back = new float[side * side];
        for (int i = 0; i < back.length; i++) {
            back[i] = random.nextFloat();
        }
        float[] left = back.clone();
        float[] right = back.clone();
        float[] disp = new float[side * side];
        int objects = 1 + random.nextInt(3);
        int[][] rects = new int[objects][];
        for (int o = 0; o < objects; o++) {
            int w = side / 4 + random.nextInt(side / 3);
            int h = side / 4 + random.nextInt(side / 3);
            int d = 1 + random.nextInt(maxDisparity);
            rects[o] = new int[] {d, random.nextInt(side - w + 1 - d) + d, random.nextInt(side - h + 1), w, h};
        }
        java.util.Arrays.sort(rects, (a, b) -> Integer.compare(a[0], b[0]));
        for (int[] q : rects) {
            int d = q[0];
            float[] tex = new float[q[3] * q[4]];
            for (int i = 0; i < tex.length; i++) {
                tex[i] = random.nextFloat();
            }
            for (int y = 0; y < q[4]; y++) {
                for (int x = 0; x < q[3]; x++) {
                    int lx = q[1] + x;
                    int ly = q[2] + y;
                    left[ly * side + lx] = tex[y * q[3] + x];
                    disp[ly * side + lx] = d;
                    right[ly * side + lx - d] = tex[y * q[3] + x];
                }
            }
        }
        for (int i = 0; i < left.length; i++) {
            left[i] += (float) (noise * random.nextGaussian());
            right[i] += (float) (noise * random.nextGaussian());
        }
        return new Stereo(side, left, right, disp);
    }

    // ------------------------------------------------------------------ videos

    /**
     * A video of moving glyph tiles.
     *
     * @param side the frame side
     * @param frames {@code [T][side * side]} frames
     * @param flow {@code [T - 1][2 * side * side]} the forward flow from frame {@code t} to
     *     {@code t + 1}: the {@code (u, v)} displacement of the pixel's content, first all
     *     {@code u}, then all {@code v}
     * @param boxes {@code [T][4 k]} the box of each object in each frame
     * @param visible {@code [T][k]} whether each object is visible (not hidden by the occluder)
     */
    public record Video(int side, float[][] frames, float[][] flow, float[][] boxes, boolean[][] visible) {

        /** @return the number of frames */
        public int length() {
            return frames.length;
        }

        /** @return the number of objects (their track ids are 0 to k - 1) */
        public int objects() {
            return boxes[0].length / 4;
        }
    }

    /**
     * Generates a video: {@code objects} textured square tiles of side {@code tile} moving at
     * constant velocities of {@code minSpeed} to {@code maxSpeed} pixels per frame in random
     * directions, bouncing off the walls, over a smooth static background. Each tile carries a
     * smooth random texture, so brightness is conserved along the motion and the flow is
     * exact. Tiles are antialiased (4x4 supersampling) so sub-pixel motion is visible. If
     * {@code occluder} is true, a static vertical band hides the objects that pass behind it
     * (for tracking under occlusion).
     *
     * @param length the number of frames
     * @param objects the number of tiles
     * @param tile the tile side in pixels
     * @param minSpeed the smallest speed
     * @param maxSpeed the largest speed
     * @param occluder whether to add the occluding band
     * @param random the random source
     * @return the video
     */
    public Video video(int length, int objects, float tile, float minSpeed, float maxSpeed, boolean occluder,
            SplittableRandom random) {
        float[] pos = new float[2 * objects];
        float[] vel = new float[2 * objects];
        float[][] tex = new float[objects][4];
        for (int o = 0; o < objects; o++) {
            pos[2 * o] = random.nextFloat() * (side - tile);
            pos[2 * o + 1] = random.nextFloat() * (side - tile);
            double a = random.nextDouble() * 2 * Math.PI;
            double s = minSpeed + (maxSpeed - minSpeed) * random.nextDouble();
            vel[2 * o] = (float) (s * Math.cos(a));
            vel[2 * o + 1] = (float) (s * Math.sin(a));
            tex[o] = new float[] {0.5f + 0.3f * random.nextFloat(), 0.6f + 0.6f * random.nextFloat(),
                0.6f + 0.6f * random.nextFloat(), (float) (random.nextDouble() * 2 * Math.PI)};
        }
        double bp = random.nextDouble() * 2 * Math.PI;
        int band = occluder ? side / 2 - 1 : -100;
        float[][] frames = new float[length][];
        float[][] flow = new float[Math.max(0, length - 1)][];
        float[][] boxes = new float[length][4 * objects];
        boolean[][] visible = new boolean[length][objects];
        for (int t = 0; t < length; t++) {
            float[] f = new float[side * side];
            int[] owner = new int[side * side];
            java.util.Arrays.fill(owner, -1);
            for (int y = 0; y < side; y++) {
                for (int x = 0; x < side; x++) {
                    f[y * side + x] = (float) (0.15 + 0.08 * Math.sin(0.5 * x + bp) * Math.cos(0.4 * y));
                }
            }
            for (int o = 0; o < objects; o++) {
                float ox = pos[2 * o];
                float oy = pos[2 * o + 1];
                boxes[t][4 * o] = ox;
                boxes[t][4 * o + 1] = oy;
                boxes[t][4 * o + 2] = ox + tile;
                boxes[t][4 * o + 3] = oy + tile;
                float cx = ox + tile / 2;
                visible[t][o] = !(occluder && cx > band - 1 && cx < band + 3);
                int x0 = Math.max(0, (int) Math.floor(ox));
                int y0 = Math.max(0, (int) Math.floor(oy));
                int x1 = Math.min(side - 1, (int) Math.ceil(ox + tile));
                int y1 = Math.min(side - 1, (int) Math.ceil(oy + tile));
                for (int y = y0; y <= y1; y++) {
                    for (int x = x0; x <= x1; x++) {
                        double cover = 0;
                        double value = 0;
                        for (int sy = 0; sy < 4; sy++) {
                            for (int sx = 0; sx < 4; sx++) {
                                double px = x + (sx + 0.5) / 4;
                                double py = y + (sy + 0.5) / 4;
                                double u = px - ox;
                                double v = py - oy;
                                if (u >= 0 && u < tile && v >= 0 && v < tile) {
                                    cover += 1 / 16.0;
                                    value += (tex[o][0] + 0.3 * Math.sin(tex[o][1] * u + tex[o][3])
                                            * Math.cos(tex[o][2] * v)) / 16.0;
                                }
                            }
                        }
                        if (cover > 0) {
                            int i = y * side + x;
                            f[i] = (float) (f[i] * (1 - cover) + value);
                            if (cover >= 0.5) {
                                owner[i] = o;
                            }
                        }
                    }
                }
            }
            if (occluder) {
                for (int y = 0; y < side; y++) {
                    for (int x = band; x < band + 2; x++) {
                        f[y * side + x] = 0.05f;
                        owner[y * side + x] = -1;
                    }
                }
            }
            if (t + 1 < length) {
                float[] fl = new float[2 * side * side];
                float[] nextPos = pos.clone();
                float[] nextVel = vel.clone();
                advance(nextPos, nextVel, tile);
                for (int i = 0; i < side * side; i++) {
                    if (owner[i] >= 0) {
                        fl[i] = nextPos[2 * owner[i]] - pos[2 * owner[i]];
                        fl[side * side + i] = nextPos[2 * owner[i] + 1] - pos[2 * owner[i] + 1];
                    }
                }
                flow[t] = fl;
            }
            for (int i = 0; i < f.length; i++) {
                f[i] += (float) (noise * random.nextGaussian());
            }
            frames[t] = f;
            advance(pos, vel, tile);
        }
        return new Video(side, frames, flow, boxes, visible);
    }

    private void advance(float[] pos, float[] vel, float tile) {
        for (int i = 0; i < pos.length; i++) {
            pos[i] += vel[i];
            if (pos[i] < 0) {
                pos[i] = -pos[i];
                vel[i] = -vel[i];
            } else if (pos[i] > side - tile) {
                pos[i] = 2 * (side - tile) - pos[i];
                vel[i] = -vel[i];
            }
        }
    }
}
