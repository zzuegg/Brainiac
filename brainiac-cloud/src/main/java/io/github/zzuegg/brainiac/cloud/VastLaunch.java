package io.github.zzuegg.brainiac.cloud;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * What to start on a rented machine.
 *
 * <p>Brainiac's GPU backend is Vulkan, not CUDA. NVIDIA's container runtime only mounts the
 * driver's Vulkan (graphics) libraries when the container asks for them, so the defaults set
 * {@code NVIDIA_DRIVER_CAPABILITIES=all}; {@code cloud/vast/setup.sh} then checks that a
 * Vulkan device really shows up, since some hosts still hand out only CUDA.
 *
 * @param image the Docker image
 * @param diskGb the disk to allocate
 * @param env environment variables for the container
 * @param label a label to find the instance by in Vast's console
 */
public record VastLaunch(String image, int diskGb, Map<String, String> env, String label) {

    /** The default image: plain Ubuntu with NVIDIA's CUDA base layer. */
    public static final String DEFAULT_IMAGE = "nvidia/cuda:12.6.3-base-ubuntu24.04";

    /**
     * Returns the defaults.
     *
     * @return {@link #DEFAULT_IMAGE}, 40 GB of disk, every driver capability, label {@code brainiac}
     */
    public static VastLaunch defaults() {
        return new VastLaunch(DEFAULT_IMAGE, 40,
                Map.of("NVIDIA_DRIVER_CAPABILITIES", "all", "NVIDIA_VISIBLE_DEVICES", "all"), "brainiac");
    }

    /**
     * Returns this launch with another image.
     *
     * @param name the Docker image
     * @return the changed launch
     */
    public VastLaunch withImage(String name) {
        return new VastLaunch(name, diskGb, env, label);
    }

    /**
     * Returns this launch with another disk size.
     *
     * @param gb the disk to allocate
     * @return the changed launch
     */
    public VastLaunch withDisk(int gb) {
        return new VastLaunch(image, gb, env, label);
    }

    /**
     * Returns this launch with container ports reachable from outside through Vast's port
     * mapping ({@link VastInstance#url} gives the address). Traffic to them is not encrypted;
     * an SSH tunnel is the safer way in.
     *
     * @param containerPorts ports inside the container, such as 7870 for a studio worker
     * @return the changed launch
     */
    public VastLaunch exposePorts(int... containerPorts) {
        Map<String, String> e = new LinkedHashMap<>(env);
        for (int p : containerPorts) {
            // Vast reads Docker options from the env map: "-p 7870:7870" -> "1"
            e.put("-p " + p + ":" + p, "1");
        }
        return new VastLaunch(image, diskGb, Map.copyOf(e), label);
    }

    /**
     * Builds the body of Vast's rent request ({@code PUT /api/v0/asks/{offer}/}).
     *
     * @return the request as JSON values
     */
    Map<String, Object> toJson() {
        Map<String, Object> b = new LinkedHashMap<>();
        b.put("client_id", "me");
        b.put("image", image);
        b.put("disk", diskGb);
        b.put("env", new LinkedHashMap<String, Object>(env));
        // SSH through Vast's proxy; the account's SSH keys are installed in the container
        b.put("runtype", "ssh_proxy");
        b.put("label", label);
        return b;
    }
}
