package io.github.zzuegg.brainiac.cloud;

import io.github.zzuegg.brainiac.io.Json;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;

/**
 * A rented Vast.ai instance.
 *
 * @param id the instance id, which {@link VastClient#destroy} takes
 * @param status the container's state: {@code created}, {@code loading}, {@code running},
 *        {@code exited}, or empty while the host is still setting it up
 * @param statusMessage the host's last message, such as an image pull error
 * @param gpuName the GPU model
 * @param dollarsPerHour what it costs per hour
 * @param sshHost the host to SSH to (Vast's SSH proxy), empty until it is known
 * @param sshPort the port on {@code sshHost}
 * @param label the label it was rented with
 * @param publicIp the machine's public address, empty until it is known
 * @param ports container ports exposed with {@link VastLaunch#exposePorts}, mapped to the
 *        ports they are reachable on at {@code publicIp}
 */
public record VastInstance(long id, String status, String statusMessage, String gpuName, double dollarsPerHour,
        String sshHost, int sshPort, String label, String publicIp, Map<Integer, Integer> ports) {

    /**
     * Reads an instance from the API's JSON.
     *
     * @param o one instance object
     * @return the instance
     */
    static VastInstance from(Map<String, Object> o) {
        return new VastInstance(
                Json.integer(o, "id"),
                Json.string(o, "actual_status", ""),
                Json.string(o, "status_msg", ""),
                Json.string(o, "gpu_name", "?"),
                Json.number(o, "dph_total", 0),
                Json.string(o, "ssh_host", ""),
                (int) Json.integer(o, "ssh_port", 0),
                Json.string(o, "label", ""),
                Json.string(o, "public_ipaddr", "").strip(),
                ports(o.get("ports")));
    }

    /** Reads Docker's port map: {@code {"7870/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41234"}]}}. */
    private static Map<Integer, Integer> ports(Object value) {
        Map<Integer, Integer> out = new TreeMap<>();
        if (!(value instanceof Map<?, ?> m)) {
            return out;
        }
        for (Map.Entry<?, ?> e : m.entrySet()) {
            String inside = String.valueOf(e.getKey()).replace("/tcp", "");
            if (!(e.getValue() instanceof List<?> l) || !inside.matches("\\d+")) {
                continue;
            }
            for (Object b : l) {
                if (b instanceof Map<?, ?> binding && binding.get("HostPort") != null
                        && String.valueOf(binding.get("HostPort")).matches("\\d+")) {
                    out.put(Integer.parseInt(inside), Integer.parseInt(String.valueOf(binding.get("HostPort"))));
                    break;
                }
            }
        }
        return Map.copyOf(out);
    }

    /**
     * Returns where a container port is reachable from outside.
     *
     * @param containerPort the port inside the container, as given to {@link VastLaunch#exposePorts}
     * @return {@code http://ip:port}, if Vast has mapped it
     */
    public Optional<String> url(int containerPort) {
        Integer p = ports.get(containerPort);
        return p == null || publicIp.isEmpty() ? Optional.empty() : Optional.of("http://" + publicIp + ":" + p);
    }

    /**
     * Returns whether the container runs and can be reached over SSH.
     *
     * @return true once it is running with an SSH address
     */
    public boolean ready() {
        return "running".equals(status) && !sshHost.isEmpty() && sshPort > 0;
    }

    @Override
    public String toString() {
        return String.format(java.util.Locale.ROOT, "#%d %s %s, $%.3f/h%s%s", id, gpuName,
                status.isEmpty() ? "starting" : status, dollarsPerHour, label.isEmpty() ? "" : ", " + label,
                statusMessage.isBlank() ? "" : " (" + statusMessage.strip() + ")");
    }
}
