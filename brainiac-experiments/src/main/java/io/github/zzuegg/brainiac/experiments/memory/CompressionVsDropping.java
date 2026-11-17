package io.github.zzuegg.brainiac.experiments.memory;

import io.github.zzuegg.brainiac.experimental.nn.CompressiveKvMemory;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Experiment X2: what happens to key/value pairs that fall out of a small cache window? A
 * stream of random key/value pairs flows through a {@link CompressiveKvMemory} in each
 * {@link CompressiveKvMemory.Mode}; afterwards every evicted key is used as a query and the
 * answer is compared with the value it was stored with. Dropping recalls nothing (cosine near
 * 0), the linear and delta memories recall a useful share at a constant memory cost, pooling
 * keeps a blurred trace.
 */
public final class CompressionVsDropping {

    /**
     * The outcome for one mode.
     *
     * @param recall the mean cosine between the answer for an evicted key and its value
     * @param memoryFloats the floats held besides the window
     */
    public record Result(double recall, long memoryFloats) {
    }

    private CompressionVsDropping() {
    }

    /**
     * Runs the experiment.
     *
     * @param pairs the pairs streamed through
     * @param window the cache window
     * @param keySize the key size; recall improves with it, as keys interfere less
     * @param seed the random seed
     * @return the result per mode name: DROP, LINEAR, DELTA, POOL, and LINEAR_DPFP and
     *     DELTA_DPFP with the DPFP feature map
     */
    public static Map<String, Result> run(int pairs, int window, int keySize, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int valueSize = 16;
        Tensor keys = Tensor.normal(Shape.of(1, pairs, keySize), 0, 1, random);
        Tensor values = Tensor.normal(Shape.of(1, pairs, valueSize), 0, 1, random);
        int evicted = pairs - window;
        Tensor oldKeys = keys.slice(1, 0, evicted);
        float[] truth = values.slice(1, 0, evicted).toArray();
        Map<String, Result> out = new LinkedHashMap<>();
        for (String name : new String[] {"DROP", "LINEAR", "DELTA", "POOL", "LINEAR_DPFP", "DELTA_DPFP"}) {
            CompressiveKvMemory.Mode mode = CompressiveKvMemory.Mode.valueOf(name.replace("_DPFP", ""));
            CompressiveKvMemory memory = CompressiveKvMemory.create(1, keySize, valueSize, window, mode)
                    .pooling(4, pairs).gate(0.5f);
            if (name.endsWith("_DPFP")) {
                memory.featureMap(CompressiveKvMemory.FeatureMap.DPFP);
            }
            memory.append(keys, values);
            float[] answer = memory.attend(oldKeys, (float) (1 / Math.sqrt(keySize))).toArray();
            double sum = 0;
            for (int i = 0; i < evicted; i++) {
                sum += cosine(answer, truth, i * valueSize, valueSize);
            }
            out.put(name, new Result(sum / evicted, memory.memoryFloats()));
        }
        return out;
    }

    private static double cosine(float[] a, float[] b, int off, int n) {
        double ab = 0;
        double aa = 0;
        double bb = 0;
        for (int i = off; i < off + n; i++) {
            ab += a[i] * b[i];
            aa += a[i] * a[i];
            bb += b[i] * b[i];
        }
        return ab / Math.sqrt(aa * bb + 1e-12);
    }
}
