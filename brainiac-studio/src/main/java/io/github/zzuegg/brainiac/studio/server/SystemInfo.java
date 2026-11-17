package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.tensor.Device;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** What this machine offers: devices, memory and the Brainiac modules that are present. */
final class SystemInfo {

    private SystemInfo() {
    }

    static Map<String, Object> snapshot() {
        Map<String, Object> out = new LinkedHashMap<>();
        List<Map<String, Object>> devices = new ArrayList<>();
        List<Device> available;
        try {
            available = Device.available();
        } catch (RuntimeException | LinkageError e) {
            available = List.of();
        }
        for (Device d : available) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("name", d.name());
            m.put("kind", d.kind().name().toLowerCase(Locale.ROOT));
            devices.add(m);
        }
        out.put("devices", devices);

        Runtime rt = Runtime.getRuntime();
        Map<String, Object> jvm = new LinkedHashMap<>();
        jvm.put("java", Runtime.version().toString());
        jvm.put("processors", rt.availableProcessors());
        jvm.put("heapUsed", rt.totalMemory() - rt.freeMemory());
        jvm.put("heapMax", rt.maxMemory());
        out.put("jvm", jvm);

        List<String> modules = new ArrayList<>();
        for (Module m : CoreCapabilities.layer().modules()) {
            if (m.getName().startsWith("io.github.zzuegg.brainiac")) {
                modules.add(m.getName());
            }
        }
        modules.sort(null);
        out.put("modules", modules);
        return out;
    }
}
