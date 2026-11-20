package io.github.zzuegg.brainiac.backend.hip;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Device;

import java.util.regex.Pattern;

import org.junit.jupiter.api.Test;

/** The parts of the backend that run on the host, so they are tested on every machine. */
class HipHostTest {

    @Test
    void rocblasIsRefusedWithoutKernelsForTheGpu(@org.junit.jupiter.api.io.TempDir java.nio.file.Path dir)
            throws java.io.IOException {
        java.nio.file.Path lib = java.nio.file.Files.createDirectories(dir.resolve("rocblas").resolve("library"));
        java.nio.file.Files.writeString(lib.resolve("TensileLibrary_lazy_gfx1100.dat"), "");
        java.nio.file.Path dll = dir.resolve("rocblas.dll");
        Rocblas.checkKernelsFor(dll, "gfx1100");
        Rocblas.checkKernelsFor(dll, "gfx1100:xnack-");
        HipException e = org.junit.jupiter.api.Assertions.assertThrows(HipException.class,
                () -> Rocblas.checkKernelsFor(dll, "gfx1201"));
        assertTrue(e.getMessage().contains("gfx1201"), e.getMessage());
        // no kernel folder to look in: rocBLAS decides
        Rocblas.checkKernelsFor(dir.resolve("elsewhere").resolve("rocblas.dll"), "gfx1201");
    }

    @Test
    void withoutADriverTheBackendIsUnavailableAndSaysWhy() {
        HipBackend b = new HipBackend();
        if (b.isAvailable()) {
            assertTrue(b.status().contains("hipRTC"), b.status());
            return;
        }
        assertTrue(b.status().startsWith("unavailable: "), b.status());
        assertFalse(b.supports(io.github.zzuegg.brainiac.tensor.DType.FLOAT32));
        // the other devices are still there
        assertTrue(Device.available().stream().noneMatch(d -> d.name().startsWith("hip")));
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
        assertEquals(256, HipContext.sizeClass(1));
        assertEquals(4096, HipContext.sizeClass(4000));
        assertEquals(5120, HipContext.sizeClass(4097));
        for (long b = 1; b < (1L << 34); b = b * 3 + 7) {
            long c = HipContext.sizeClass(b);
            assertTrue(c >= b && c <= Math.max(256, b * 1.25 + 256), b + " -> " + c);
            assertEquals(c, HipContext.sizeClass(c), "a class is its own class");
        }
    }

    @Test
    void newestLibraryVersionWins() {
        Pattern linux = Pattern.compile("libhiprtc\\.so\\.([0-9.]+)");
        assertTrue(Binder.version(linux, "libhiprtc.so.7.2.26015") > Binder.version(linux, "libhiprtc.so.7"));
        assertTrue(Binder.version(linux, "libhiprtc.so.7") > Binder.version(linux, "libhiprtc.so.6.4.43482"));
        Pattern windows = Pattern.compile("hiprtc(\\d+)\\.dll");
        assertTrue(Binder.version(windows, "hiprtc0702.dll") > Binder.version(windows, "hiprtc0604.dll"));
        // without the HIP SDK the list may be empty
        for (java.nio.file.Path folder : Binder.toolkitFolders()) {
            assertTrue(folder.isAbsolute() || folder.getRoot() == null, folder.toString());
        }
    }

    /** Kernels compile in parts, each at first use; every name the backend launches has its part. */
    @Test
    void kernelsCompileInParts() throws java.io.IOException {
        assertEquals("CORE", HipContext.part("binary_contiguous_f16"));
        assertEquals("CORE", HipContext.part("random_values"));
        assertEquals("FEW_BF16", HipContext.part("few_dense_bf16"));
        assertEquals("FEW_F32", HipContext.part("unpack_packed_f32"));
        assertEquals("FEW_F16", HipContext.part("few_sum_f16"));
        assertEquals("ATTENTION_F16", HipContext.part("attention_combine_f16"));
        assertEquals("FFT", HipContext.part("fft_rows"));
        String source;
        try (java.io.InputStream in = HipContext.class.getResourceAsStream("kernels.hip")) {
            source = new String(in.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
        for (String part : new String[] {"CORE", "FEW_F32", "FEW_F64", "FEW_F16", "FEW_BF16", "ATTENTION_F32",
            "ATTENTION_F16", "ATTENTION_BF16", "FFT"}) {
            assertTrue(source.contains("#ifdef BRAINIAC_" + part + "\n"), part);
        }
    }
}
