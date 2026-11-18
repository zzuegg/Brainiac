package io.github.zzuegg.brainiac.experimental.events;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A batch of irregularly observed series on a shared time grid.
 *
 * @param values {@code [batch, steps, channels]}, 0 where not observed
 * @param mask {@code [batch, steps, channels]}, 1 where observed
 * @param delta time since the channel's previous observation (GRU-D's {@code delta})
 * @param last the channel's most recent observed value strictly before each step (0 if none)
 * @param times the grid times
 */
public record IrregularBatch(Tensor values, Tensor mask, Tensor delta, Tensor last, double[] times) {

    /**
     * Builds a batch.
     *
     * @param values complete or partially known values {@code [series][steps][channels]}
     * @param observed which values the model may see
     * @param rows the series in the batch
     * @param times the grid
     * @return the batch
     */
    public static IrregularBatch of(float[][][] values, boolean[][][] observed, int[] rows, double[] times) {
        int b = rows.length;
        int steps = times.length;
        int c = values[0][0].length;
        float[] v = new float[b * steps * c];
        float[] m = new float[b * steps * c];
        float[] d = new float[b * steps * c];
        float[] l = new float[b * steps * c];
        for (int i = 0; i < b; i++) {
            for (int ch = 0; ch < c; ch++) {
                double lastTime = times[0];
                float lastValue = 0;
                for (int t = 0; t < steps; t++) {
                    int k = (i * steps + t) * c + ch;
                    d[k] = (float) (times[t] - lastTime);
                    l[k] = lastValue;
                    if (observed[rows[i]][t][ch]) {
                        v[k] = values[rows[i]][t][ch];
                        m[k] = 1;
                        lastTime = times[t];
                        lastValue = v[k];
                    }
                }
            }
        }
        Shape s = Shape.of(b, steps, c);
        return new IrregularBatch(Tensor.of(s, v), Tensor.of(s, m), Tensor.of(s, d), Tensor.of(s, l), times);
    }

    /** @return the batch size */
    public int batch() {
        return values.shape().dim(0);
    }

    /** @return the grid length */
    public int steps() {
        return values.shape().dim(1);
    }

    /** @return the channel count */
    public int channels() {
        return values.shape().dim(2);
    }
}
