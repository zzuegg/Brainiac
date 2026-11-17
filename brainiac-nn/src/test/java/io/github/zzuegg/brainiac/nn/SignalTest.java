package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SignalTest {

    private static float[] tone(int n, double freq, int rate) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) Math.sin(2 * Math.PI * freq * i / rate);
        }
        return x;
    }

    @Test
    void stftFindsATonesBinAndIstftInvertsIt() {
        int rate = 8000;
        float[] x = tone(2000, 1000, rate);
        Var spec = Signal.stft(Var.of(Tensor.of(x)), 256, 64);
        assertEquals(Shape.of(2000 / 64 + 1, 129, 2), spec.shape());
        float[] power = Signal.power(spec).value().toArray();
        int frame = 10;
        int best = 0;
        for (int b = 0; b < 129; b++) {
            if (power[frame * 129 + b] > power[frame * 129 + best]) {
                best = b;
            }
        }
        assertEquals(1000 * 256 / rate, best);   // bin 32
        float[] back = Signal.istft(spec, 256, 64, 2000).value().toArray();
        assertArrayEquals(x, back, 1e-4f);
    }

    @Test
    void melFilterbankAndLogMel() {
        Tensor fb = Signal.melFilterbank(16000, 512, 40, 0, 8000);
        assertEquals(Shape.of(257, 40), fb.shape());
        float[] f = fb.toArray();
        for (int m = 0; m < 40; m++) {
            float max = 0;
            for (int b = 0; b < 257; b++) {
                max = Math.max(max, f[b * 40 + m]);
                assertTrue(f[b * 40 + m] >= 0);
            }
            assertTrue(max > 0.3f, "filter " + m + " peaks at " + max);
        }
        Var mel = Signal.logMel(Var.of(Tensor.of(Shape.of(2, 4000), new float[8000])), 16000, 512, 160, 40);
        assertEquals(Shape.of(2, 4000 / 160 + 1, 40), mel.shape());
    }

    @Test
    void stftHasGradients() {
        Tensor x = Tensor.normal(Shape.of(64), 0, 1, new SplittableRandom(1)).to(DType.FLOAT64);
        GradCheck.Result r = GradCheck.check(v -> Signal.power(Signal.stft(v.get(0), 16, 4)).sum(), List.of(x));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void specAugmentZeroesBandsAndResampleKeepsATone() {
        Var spec = Var.of(Tensor.full(Shape.of(20, 10), 1f));
        float[] m = Signal.specAugment(spec, 1, 3, 1, 5, new SplittableRandom(3)).value().toArray();
        int zeros = 0;
        for (float v : m) {
            zeros += v == 0 ? 1 : 0;
        }
        assertTrue(zeros > 0 && zeros < 200);
        float[] x = tone(1600, 440, 16000);
        float[] down = Signal.resample(x, 16000, 8000);
        assertEquals(800, down.length);
        float[] expected = tone(800, 440, 8000);
        for (int i = 100; i < 700; i++) {
            assertEquals(expected[i], down[i], 0.02f, "sample " + i);
        }
        assertEquals(Device.cpu(), Signal.hann(4, Device.cpu()).device());
    }
}
