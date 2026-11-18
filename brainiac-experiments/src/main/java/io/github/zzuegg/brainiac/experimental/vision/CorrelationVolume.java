package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * The all-pairs correlation volume of RAFT (Teed and Deng, 2020): the dot product of every
 * feature vector of the first frame with every feature vector of the second, a 4-D volume
 * {@code [h, w, h, w]} per image, average-pooled over its last two dimensions into a small
 * pyramid. {@link #lookup} reads, for each pixel and its current flow estimate, the
 * correlations in a {@code (2r+1)^2} window around where the pixel is thought to have moved,
 * on every pyramid level, so coarse levels see large displacements and fine levels precise
 * ones. Lookups are bilinear ({@link AffineSampler#sample}); gradients flow into the
 * features, not into the sampling locations (as in RAFT, which detaches the coordinates).
 *
 * <p>Paper: Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
 * (2020), <a href="https://arxiv.org/abs/2003.12039">arXiv:2003.12039</a>.
 */
public final class CorrelationVolume {

    private final int batch;
    private final int height;
    private final int width;
    private final List<Var> pyramid = new ArrayList<>();

    /**
     * Builds the volume and its pyramid.
     *
     * @param first features of the first frame, {@code [batch, c, h, w]}
     * @param second features of the second frame, same shape
     * @param levels the pyramid levels
     */
    public CorrelationVolume(Var first, Var second, int levels) {
        this.batch = first.shape().dim(0);
        int c = first.shape().dim(1);
        this.height = first.shape().dim(2);
        this.width = first.shape().dim(3);
        int hw = height * width;
        Var a = first.reshape(batch, c, hw).permute(0, 2, 1);
        Var b = second.reshape(batch, c, hw);
        Var corr = a.matmul(b).mul((float) (1 / Math.sqrt(c))).reshape(batch * hw, 1, height, width);
        pyramid.add(corr);
        for (int l = 1; l < levels; l++) {
            corr = Functional.pool(corr, Functional.PoolKind.AVG, new int[] {2, 2}, new int[] {2, 2}, new int[] {0, 0},
                    new int[] {0, 0}, new int[] {1, 1}, false, false, 1f);
            pyramid.add(corr);
        }
    }

    /** @return the pyramid levels */
    public int levels() {
        return pyramid.size();
    }

    /**
     * Looks up correlations around target locations.
     *
     * @param targets per image and pixel the location {@code (x, y)} in the second frame's
     *     pixel grid where the pixel is thought to be: {@code [batch, 2, h, w]} values (all
     *     x, then all y, per image)
     * @param radius the window radius
     * @return {@code [batch, levels * (2 radius + 1)^2, h, w]}
     */
    public Var lookup(float[] targets, int radius) {
        int hw = height * width;
        int k = 2 * radius + 1;
        List<Var> parts = new ArrayList<>();
        for (int l = 0; l < pyramid.size(); l++) {
            Var level = pyramid.get(l);
            int lh = level.shape().dim(2);
            int lw = level.shape().dim(3);
            float scale = 1f / (1 << l);
            float[] grid = new float[batch * hw * k * k * 2];
            for (int b = 0; b < batch; b++) {
                for (int p = 0; p < hw; p++) {
                    float cx = targets[(b * 2) * hw + p] * scale;
                    float cy = targets[(b * 2 + 1) * hw + p] * scale;
                    for (int i = 0; i < k; i++) {
                        for (int j = 0; j < k; j++) {
                            int o = (((b * hw + p) * k + i) * k + j) * 2;
                            grid[o] = lw == 1 ? 0 : 2 * (cx + j - radius) / (lw - 1) - 1;
                            grid[o + 1] = lh == 1 ? 0 : 2 * (cy + i - radius) / (lh - 1) - 1;
                        }
                    }
                }
            }
            Var s = AffineSampler.sample(level, Var.of(Tensor.of(Shape.of(batch * hw, k, k, 2), grid)));
            parts.add(s.reshape(batch, height, width, k * k).permute(0, 3, 1, 2));
        }
        return Var.concat(1, parts.toArray(new Var[0]));
    }
}
