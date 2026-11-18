package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.experimental.rl.KeyDoorGrid;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Renders a small environment's state as an image, so the same task can be learned from
 * pixels instead of from its compact state: the "Atari in miniature" set-up of the
 * sample-efficiency recipes (SPR, DrQ, EfficientZero). The image is
 * {@code [channels, height, width]} flattened row-major, values in {@code [0, 1]}; the wrapped
 * environment is not changed.
 *
 * <ul>
 *   <li>{@link #grid}: a {@link GridWorld} as three channels (walls, goal and hazards, agent),
 *       every cell {@code scale x scale} pixels.</li>
 *   <li>{@link #keyDoor}: a {@link KeyDoorGrid}'s three channels (agent, key, door) upscaled.</li>
 *   <li>{@link #pendulum}: a {@link Pendulum}'s rod drawn on a square canvas, with the previous
 *       frame as a second channel so the speed can be read off the image.</li>
 * </ul>
 */
public final class PixelObservations implements Environment {

    private interface Renderer {
        float[] render(float[] observation, float[] previous);
    }

    private final Environment inner;
    private final Renderer renderer;
    private final int channels;
    private final int height;
    private final int width;
    private float[] last;

    private PixelObservations(Environment inner, Renderer renderer, int channels, int height, int width) {
        this.inner = inner;
        this.renderer = renderer;
        this.channels = channels;
        this.height = height;
        this.width = width;
    }

    /**
     * A grid world drawn as an image. The grid must use one-hot (not coordinate) observations.
     *
     * @param env the grid
     * @param scale pixels per cell side
     * @return the pixel environment
     */
    public static PixelObservations grid(GridWorld env, int scale) {
        List<String> map = env.map();
        int rows = map.size();
        int cols = map.get(0).length();
        int h = rows * scale;
        int w = cols * scale;
        float[] background = new float[3 * h * w];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                char ch = map.get(r).charAt(c);
                int channel = ch == '#' ? 0 : ch == 'G' || ch == 'H' || ch == 'C' ? 1 : -1;
                if (channel >= 0) {
                    fill(background, channel, h, w, r * scale, c * scale, scale, ch == 'H' || ch == 'C' ? 0.5f : 1f);
                }
            }
        }
        return new PixelObservations(env, (obs, prev) -> {
            float[] img = background.clone();
            int cell = 0;
            while (cell < obs.length && obs[cell] == 0) {
                cell++;
            }
            fill(img, 2, h, w, (cell / cols) * scale, (cell % cols) * scale, scale, 1f);
            return img;
        }, 3, h, w);
    }

    /**
     * A key-and-door grid drawn as an image.
     *
     * @param env the grid
     * @param size the grid side the environment was built with
     * @param scale pixels per cell side
     * @return the pixel environment
     */
    public static PixelObservations keyDoor(KeyDoorGrid env, int size, int scale) {
        int h = size * scale;
        return new PixelObservations(env, (obs, prev) -> {
            float[] img = new float[3 * h * h];
            for (int c = 0; c < 3; c++) {
                for (int cell = 0; cell < size * size; cell++) {
                    float v = obs[c * size * size + cell];
                    if (v != 0) {
                        fill(img, c, h, h, (cell / size) * scale, (cell % size) * scale, scale, v);
                    }
                }
            }
            return img;
        }, 3, h, h);
    }

    /**
     * A pendulum drawn as a rod from the canvas centre, with the previous frame as a second
     * channel.
     *
     * @param env the pendulum
     * @param size the canvas side in pixels
     * @return the pixel environment
     */
    public static PixelObservations pendulum(Pendulum env, int size) {
        return new PixelObservations(env, (obs, prev) -> {
            float[] img = new float[2 * size * size];
            rod(img, 0, size, obs[0], obs[1]);
            float[] p = prev == null ? obs : prev;
            rod(img, 1, size, p[0], p[1]);
            return img;
        }, 2, size, size);
    }

    private static void rod(float[] img, int channel, int size, float cos, float sin) {
        double c = (size - 1) / 2.0;
        double len = c;
        for (int k = 0; k <= 2 * size; k++) {
            double t = k / (2.0 * size);
            // upright (angle 0) points up: x = sin, y = -cos
            int x = (int) Math.round(c + t * len * sin);
            int y = (int) Math.round(c - t * len * cos);
            if (x >= 0 && x < size && y >= 0 && y < size) {
                img[channel * size * size + y * size + x] = 1;
            }
        }
    }

    private static void fill(float[] img, int channel, int h, int w, int top, int left, int scale, float value) {
        for (int y = top; y < top + scale; y++) {
            for (int x = left; x < left + scale; x++) {
                img[channel * h * w + y * w + x] = value;
            }
        }
    }

    /** @return the image channels */
    public int channels() {
        return channels;
    }

    /** @return the image height */
    public int height() {
        return height;
    }

    /** @return the image width */
    public int width() {
        return width;
    }

    /** @return the wrapped environment */
    public Environment inner() {
        return inner;
    }

    @Override
    public Space observationSpace() {
        return Space.box(channels * height * width, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return inner.actionSpace();
    }

    @Override
    public float[] reset(SplittableRandom random) {
        float[] obs = inner.reset(random);
        last = obs;
        return renderer.render(obs, null);
    }

    @Override
    public Step step(float[] action) {
        Step s = inner.step(action);
        float[] img = renderer.render(s.observation(), last);
        last = s.observation();
        return new Step(img, s.reward(), s.terminated(), s.truncated());
    }

    @Override
    public String render() {
        return inner.render();
    }

    @Override
    public String toString() {
        return "Pixels(" + inner + ", " + channels + "x" + height + "x" + width + ")";
    }
}
