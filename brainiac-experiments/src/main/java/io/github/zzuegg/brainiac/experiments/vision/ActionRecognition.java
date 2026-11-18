package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.experimental.data.ShapeClips;
import io.github.zzuegg.brainiac.experimental.vision.VideoClassifiers;

import java.util.ArrayList;
import java.util.List;

/**
 * Wave 11 item 11.5.13: action recognition on {@link ShapeClips} (16 px, 8 frames, eight actions
 * in four time-reversed pairs). Every {@link VideoClassifiers.Kind}: accuracy, direction
 * accuracy (correct among clips whose predicted action is in the right pair; 0.5 is chance for
 * an order-blind model) and multiply-adds per clip.
 */
public final class ActionRecognition {

    private ActionRecognition() {
    }

    /**
     * A row.
     *
     * @param method the architecture
     * @param accuracy test accuracy
     * @param direction direction accuracy within the right pair
     * @param macs multiply-adds per clip
     */
    public record Row(String method, double accuracy, double direction, long macs) {
    }

    /**
     * Runs the comparison.
     *
     * @param train training clips
     * @param test test clips
     * @param steps steps per model
     * @param seed the seed
     * @return rows
     */
    public static List<Row> run(int train, int test, int steps, long seed) {
        ShapeClips gen = new ShapeClips(16, 8, 0.1f);
        List<ShapeClips.Clip> tr = gen.generate(train, seed);
        List<ShapeClips.Clip> te = gen.generate(test, seed + 1);
        List<float[]> x = new ArrayList<>();
        int[] y = new int[tr.size()];
        for (int i = 0; i < tr.size(); i++) {
            x.add(tr.get(i).frames());
            y[i] = tr.get(i).action();
        }
        List<Row> rows = new ArrayList<>();
        for (VideoClassifiers.Kind k : VideoClassifiers.Kind.values()) {
            VideoClassifiers m = new VideoClassifiers(k, 6, ShapeClips.ACTIONS, seed);
            m.train(x, y, 8, 16, steps, 8, 4e-3f, seed);
            int ok = 0;
            int pair = 0;
            int dir = 0;
            for (ShapeClips.Clip c : te) {
                int p = m.predict(c.frames(), 8, 16);
                ok += p == c.action() ? 1 : 0;
                if (p / 2 == c.action() / 2) {
                    pair++;
                    dir += p == c.action() ? 1 : 0;
                }
            }
            rows.add(new Row(k.name(), ok / (double) te.size(), dir / (double) Math.max(1, pair), m.macs(8, 16)));
        }
        return rows;
    }
}
