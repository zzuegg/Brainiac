package io.github.zzuegg.brainiac.cloud;

import io.github.zzuegg.brainiac.io.Json;

import java.util.Map;

/**
 * A machine on Vast.ai that can be rented, as the offer search returns it.
 *
 * @param id the offer id, which {@link VastClient#rent} takes
 * @param gpuName the GPU model, such as {@code RTX 4090}
 * @param gpus how many GPUs come with it
 * @param gpuMemoryGb memory per GPU, in GB
 * @param dollarsPerHour the total price per hour (GPU, storage and the default disk)
 * @param reliability the host's reliability score, 0 to 1
 * @param driverVersion the NVIDIA driver on the host
 * @param location where the machine is, such as {@code Germany, DE}
 * @param downloadMbps measured download speed
 * @param diskGb free disk space
 * @param cpuCores CPU cores that come with the offer
 * @param verified whether Vast has verified the host
 */
public record VastOffer(long id, String gpuName, int gpus, double gpuMemoryGb, double dollarsPerHour,
        double reliability, String driverVersion, String location, double downloadMbps, double diskGb,
        double cpuCores, boolean verified) {

    /**
     * Reads an offer from the search's JSON.
     *
     * @param o one entry of the {@code offers} array
     * @return the offer
     */
    static VastOffer from(Map<String, Object> o) {
        return new VastOffer(
                Json.integer(o, "id"),
                Json.string(o, "gpu_name", "?"),
                (int) Json.integer(o, "num_gpus", 1),
                Json.number(o, "gpu_ram", 0) / 1024.0,
                Json.number(o, "dph_total", 0),
                Json.number(o, "reliability2", Json.number(o, "reliability", 0)),
                Json.string(o, "driver_version", "?"),
                Json.string(o, "geolocation", "?"),
                Json.number(o, "inet_down", 0),
                Json.number(o, "disk_space", 0),
                Json.number(o, "cpu_cores_effective", Json.number(o, "cpu_cores", 0)),
                Boolean.TRUE.equals(o.get("verified")) || "verified".equals(o.get("verification")));
    }

    @Override
    public String toString() {
        return String.format(java.util.Locale.ROOT, "#%d %dx %s %.0f GB, $%.3f/h, %s, driver %s, reliability %.3f",
                id, gpus, gpuName, gpuMemoryGb, dollarsPerHour, location, driverVersion, reliability);
    }
}
