package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.Device;

import java.util.Locale;

/**
 * The device the tests run on: the CPU unless {@code -PtestDevice=...} names another, by a
 * part of its name ({@code vulkan}, {@code opencl}, {@code 9070}, ...).
 */
final class TestDevice {

    private TestDevice() {
    }

    static Device get() {
        String want = System.getProperty("brainiac.test.device", "cpu").toLowerCase(Locale.ROOT);
        if (want.equals("cpu")) {
            return Device.cpu();
        }
        return Device.available().stream()
                .filter(d -> d.name().toLowerCase(Locale.ROOT).contains(want))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("no device matches '" + want + "': "
                        + Device.available()));
    }
}
