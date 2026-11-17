package io.github.zzuegg.brainiac.backend.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Device;

import java.util.regex.Pattern;

import org.junit.jupiter.api.Test;

/** The parts of the backend that run on the host, so they are tested on every machine. */
class CudaHostTest {

    @Test
    void withoutADriverTheBackendIsUnavailableAndSaysWhy() {
        CudaBackend b = new CudaBackend();
        if (b.isAvailable()) {
            assertTrue(b.status().contains("NVRTC"), b.status());
            return;
        }
        assertTrue(b.status().startsWith("unavailable: "), b.status());
        assertFalse(b.supports(io.github.zzuegg.brainiac.tensor.DType.FLOAT32));
        // the other devices are still there
        assertTrue(Device.available().stream().noneMatch(d -> d.name().startsWith("cuda")));
        assertEquals(Device.Kind.CPU, Device.cpu().kind());
    }

    @Test
    void axesMergeWhereEveryOperandAllows() {
        // a contiguous [2, 3, 4] for both: one axis
        int[] shape = {2, 3, 4};
        long[] c = Axes.contiguous(shape);
        Axes one = Axes.merge(shape, c, c);
        assertArrayEquals(new int[] {24}, one.shape);
        assertArrayEquals(new long[] {1}, one.strides[0]);
        // a row broadcast: the broadcast axis stays apart
        Axes rows = Axes.merge(shape, c, new long[] {0, 4, 1});
        assertArrayEquals(new int[] {2, 12}, rows.shape);
        assertArrayEquals(new long[] {0, 1}, rows.strides[1]);
        // size-1 axes drop out, and a shape of ones keeps one axis
        Axes ones = Axes.merge(new int[] {1, 1}, new long[] {1, 1}, new long[] {1, 1});
        assertArrayEquals(new int[] {1}, ones.shape);
    }

    @Test
    void ranksAboveTheLimitRunAsSeveralBlocks() {
        int[] shape = {2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        long[] in = new long[shape.length];
        long step = 1;
        for (int d = 0; d < shape.length; d++) {   // reversed strides: nothing merges
            in[d] = step;
            step *= shape[d];
        }
        Axes ax = Axes.merge(shape, in, Axes.contiguous(shape));
        assertEquals(10, ax.rank());
        long[] total = new long[1];
        int[] blocks = new int[1];
        ax.forEachBlock((offsets, inner, strides, count) -> {
            assertEquals(Axes.MAX_AXES, inner.length);
            total[0] += count;
            blocks[0]++;
        });
        assertEquals(6, blocks[0]);
        assertEquals(2L * 3 * 2 * 3 * 2 * 3 * 2 * 3 * 2 * 3, total[0]);
    }

    @Test
    void sizeClassesRoundUpALittle() {
        assertEquals(256, CudaContext.sizeClass(1));
        assertEquals(4096, CudaContext.sizeClass(4000));
        assertEquals(5120, CudaContext.sizeClass(4097));
        for (long b = 1; b < (1L << 34); b = b * 3 + 7) {
            long c = CudaContext.sizeClass(b);
            assertTrue(c >= b && c <= Math.max(256, b * 1.25 + 256), b + " -> " + c);
            assertEquals(c, CudaContext.sizeClass(c), "a class is its own class");
        }
    }

    @Test
    void newestLibraryVersionWins() {
        Pattern linux = Pattern.compile("libnvrtc\\.so\\.([0-9.]+)");
        assertTrue(Binder.version(linux, "libnvrtc.so.12.4.127") > Binder.version(linux, "libnvrtc.so.12"));
        assertTrue(Binder.version(linux, "libnvrtc.so.13") > Binder.version(linux, "libnvrtc.so.12.9.86"));
        Pattern windows = Pattern.compile("nvrtc64_(\\d+)_0\\.dll");
        assertTrue(Binder.version(windows, "nvrtc64_130_0.dll") > Binder.version(windows, "nvrtc64_120_0.dll"));
        assertFalse(Binder.toolkitFolders().isEmpty());
    }
}
