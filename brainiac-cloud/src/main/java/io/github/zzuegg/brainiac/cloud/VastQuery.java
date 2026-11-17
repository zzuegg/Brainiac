package io.github.zzuegg.brainiac.cloud;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * What to search Vast.ai's offers for. The defaults fit Brainiac, which trains on one GPU per
 * run: a single RTX 4090, RTX 5090, A100 or H100 on a verified, reliable host with a fast
 * download (the first run fetches two JDKs and Gradle), cheapest first.
 *
 * @param gpuNames GPU models as Vast names them, such as {@code RTX 4090} or {@code H100 SXM}
 * @param gpus GPUs per machine
 * @param maxDollarsPerHour the most to pay per hour; 0 for no limit
 * @param minReliability the least reliability score, 0 to 1
 * @param minDownloadMbps the least download speed
 * @param minDiskGb the least free disk
 * @param minGpuMemoryGb the least memory per GPU, which also keeps out mislisted cards (a
 *        "RTX 4090" with 16 GB is a laptop part)
 * @param limit how many offers to return
 */
public record VastQuery(List<String> gpuNames, int gpus, double maxDollarsPerHour, double minReliability,
        double minDownloadMbps, double minDiskGb, double minGpuMemoryGb, int limit) {

    /** The GPU models Brainiac is meant for, as Vast names them. */
    public static final List<String> DEFAULT_GPUS = List.of(
            "RTX 4090", "RTX 5090", "A100 PCIE", "A100 SXM4", "A100X", "H100 PCIE", "H100 SXM", "H100 NVL");

    /**
     * Returns the default search.
     *
     * @return one GPU of {@link #DEFAULT_GPUS} with at least 23 GB, reliability 0.98, 200 Mbit/s,
     *         40 GB disk, 20 offers
     */
    public static VastQuery defaults() {
        return new VastQuery(DEFAULT_GPUS, 1, 0, 0.98, 200, 40, 23, 20);
    }

    /**
     * Returns this search with another least GPU memory.
     *
     * @param gb the least memory per GPU, in GB
     * @return the changed search
     */
    public VastQuery withMinGpuMemory(double gb) {
        return new VastQuery(gpuNames, gpus, maxDollarsPerHour, minReliability, minDownloadMbps, minDiskGb, gb, limit);
    }

    /**
     * Returns this search for other GPU models.
     *
     * @param names GPU models as Vast names them
     * @return the changed search
     */
    public VastQuery withGpus(List<String> names) {
        return new VastQuery(List.copyOf(names), gpus, maxDollarsPerHour, minReliability, minDownloadMbps, minDiskGb, minGpuMemoryGb, limit);
    }

    /**
     * Returns this search with a price limit.
     *
     * @param dollars the most to pay per hour; 0 for no limit
     * @return the changed search
     */
    public VastQuery withMaxPrice(double dollars) {
        return new VastQuery(gpuNames, gpus, dollars, minReliability, minDownloadMbps, minDiskGb, minGpuMemoryGb, limit);
    }

    /**
     * Returns this search with another result count.
     *
     * @param n how many offers to return
     * @return the changed search
     */
    public VastQuery withLimit(int n) {
        return new VastQuery(gpuNames, gpus, maxDollarsPerHour, minReliability, minDownloadMbps, minDiskGb, minGpuMemoryGb, n);
    }

    /**
     * Builds the request body of Vast's offer search ({@code POST /api/v0/bundles/}).
     *
     * @return the query as JSON values
     */
    Map<String, Object> toJson() {
        Map<String, Object> q = new LinkedHashMap<>();
        q.put("verified", Map.of("eq", true));
        q.put("rentable", Map.of("eq", true));
        q.put("rented", Map.of("eq", false));
        q.put("external", Map.of("eq", false));
        q.put("num_gpus", Map.of("eq", gpus));
        if (!gpuNames.isEmpty()) {
            q.put("gpu_name", Map.of("in", new ArrayList<Object>(gpuNames)));
        }
        if (maxDollarsPerHour > 0) {
            q.put("dph_total", Map.of("lte", maxDollarsPerHour));
        }
        q.put("reliability2", Map.of("gte", minReliability));
        q.put("inet_down", Map.of("gte", minDownloadMbps));
        q.put("disk_space", Map.of("gte", minDiskGb));
        if (minGpuMemoryGb > 0) {
            // Vast keeps gpu_ram in MB
            q.put("gpu_ram", Map.of("gte", minGpuMemoryGb * 1000));
        }
        q.put("order", List.of(List.of("dph_total", "asc")));
        q.put("type", "on-demand");
        q.put("limit", limit);
        return q;
    }
}
