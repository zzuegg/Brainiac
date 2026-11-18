package io.github.zzuegg.brainiac.studio.train;

import io.github.zzuegg.brainiac.cloud.VastOffer;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ModelInfo;
import io.github.zzuegg.brainiac.studio.chat.VastAi;
import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/**
 * Runners a project's routes name, set up on demand: a GPU machine rented on Vast.ai, built into
 * a worker host, with the route's model downloaded onto it. A runner stays rented for the hours
 * it allows, so the next task reuses it; it is destroyed after them like any rental.
 *
 * <p>A route names a runner as {@code vast:<runner>}; the runner says what to rent and which
 * model to fetch. Setting one up takes 10 to 20 minutes and costs what Vast.ai bills, never more
 * an hour than the runner's {@link Runner#maxPrice()}.
 */
public final class VastRunners {

    private static final System.Logger LOG = System.getLogger(VastRunners.class.getName());

    /** How a route's fallback names a runner. */
    public static final String PREFIX = "vast:";

    /**
     * What to rent and fetch for a runner.
     *
     * @param gpu the GPU's name to look for, such as {@code 4090}, or empty for any
     * @param minGpuRam the least memory per GPU, in GB
     * @param maxPrice the most dollars an hour it may cost; must be set
     * @param hours how long it stays rented before it is destroyed
     * @param repo the Hugging Face repository of the model it runs
     * @param option {@code folder} for the whole repository, or a GGUF file in it
     * @param model the model's name on the machine, or empty for the repository's name
     */
    public record Runner(String gpu, double minGpuRam, double maxPrice, double hours, String repo, String option,
            String model) {

        /** Checks it. */
        public Runner {
            gpu = gpu == null ? "" : gpu.strip();
            repo = repo == null ? "" : repo.strip();
            option = option == null || option.isBlank() ? "folder" : option.strip();
            model = model == null ? "" : model.strip();
            if (!(maxPrice > 0)) {
                throw new IllegalArgumentException("a Vast.ai runner needs the most it may cost an hour");
            }
            if (!(hours > 0) || hours > VastRentals.MAX_HOURS) {
                throw new IllegalArgumentException("a Vast.ai runner stays rented between a few minutes and "
                        + VastRentals.MAX_HOURS + " hours");
            }
        }

        /** @return the name the model has on the machine */
        public String modelName() {
            if (!model.isEmpty()) {
                return model;
            }
            if (!"folder".equals(option)) {
                String f = option.substring(option.lastIndexOf('/') + 1);
                return f.toLowerCase(Locale.ROOT).endsWith(".gguf") ? f.substring(0, f.length() - 5) : f;
            }
            return repo.substring(repo.indexOf('/') + 1);
        }

        /**
         * Reads one from JSON.
         *
         * @param m {@code {gpu, minGpuRam, maxPrice, hours, repo, option, model}}
         * @return the runner
         */
        public static Runner fromJson(Map<String, Object> m) {
            return new Runner(Json.string(m, "gpu", ""), Json.number(m, "minGpuRam", 0), Json.number(m, "maxPrice", 0),
                    Json.number(m, "hours", 2), Json.string(m, "repo", ""), Json.string(m, "option", "folder"),
                    Json.string(m, "model", ""));
        }
    }

    private final Supplier<VastRentals> rentals;
    private final Supplier<VastAi> vast;
    private final Hosts hosts;
    private volatile long pollMillis = 5000;
    private volatile long setupMillis = 45 * 60_000L;

    /**
     * Sets runners up with these rentals.
     *
     * @param rentals the studio's rentals
     * @param vast the Vast.ai account
     * @param hosts the worker hosts rented machines join
     */
    public VastRunners(Supplier<VastRentals> rentals, Supplier<VastAi> vast, Hosts hosts) {
        this.rentals = rentals;
        this.vast = vast;
        this.hosts = hosts;
    }

    /** For tests: how often to look and how long setting up may take. */
    void timing(long pollMillis, long setupMillis) {
        this.pollMillis = pollMillis;
        this.setupMillis = setupMillis;
    }

    /**
     * Makes sure a runner is up, with its model, and names the model there.
     *
     * @param tag the runner's name in its project, such as {@code film-a/video}, to find its machine again
     * @param r what to rent and fetch
     * @param cancelled says when to stop waiting (the machine stays rented)
     * @return the model's key, {@code host-<id>/<model>}
     * @throws IOException if nothing can be rented in the price, the machine fails, or the model
     *     cannot be fetched
     * @throws InterruptedException if interrupted while waiting
     */
    public String ensure(String tag, Runner r, BooleanSupplier cancelled) throws IOException, InterruptedException {
        VastRentals vr = rentals.get();
        if (vr == null) {
            throw new IOException("renting is not set up in this studio");
        }
        long deadline = System.currentTimeMillis() + setupMillis;
        Map<String, Object> rental = find(vr, tag).orElse(null);
        if (rental == null) {
            rental = rent(vr, tag, r);
        }
        String key = String.valueOf(rental.get("key"));
        // wait until it joined as a host
        while (true) {
            Map<String, Object> now = vr.list().stream().filter(m -> key.equals(m.get("key"))).findFirst().orElse(null);
            if (now == null || "failed".equals(now.get("stage")) || "destroyed".equals(now.get("stage"))
                    || "destroying".equals(now.get("stage"))) {
                throw new IOException("the Vast.ai machine for " + tag + " did not come up"
                        + (now != null && now.get("error") != null ? ": " + now.get("error") : ""));
            }
            if ("ready".equals(now.get("stage")) && now.get("host") != null) {
                rental = now;
                break;
            }
            waitOr(cancelled, deadline, "setting up the Vast.ai machine for " + tag);
        }
        Hosts.Host host = hosts.find(String.valueOf(rental.get("host")))
                .orElseThrow(() -> new IOException("the Vast.ai machine for " + tag + " is no longer a host"));
        WorkerClient client = hosts.client(host);
        String name = r.modelName();
        if (!has(client, name)) {
            if (r.repo().isEmpty()) {
                throw new IOException("the Vast.ai machine for " + tag + " has no model " + name
                        + " and the runner names no repository to fetch it from");
            }
            LOG.log(System.Logger.Level.INFO, "fetching " + r.repo() + " onto the Vast.ai machine for " + tag);
            client.hubDownload(r.repo(), r.option());
            while (!has(client, name)) {
                Optional<Map<String, Object>> d = client.hubDownloads().stream().map(Json::object)
                        .filter(m -> r.repo().equals(m.get("repo"))).findFirst();
                if (d.isPresent() && ("failed".equals(d.get().get("state")) || "stopped".equals(d.get().get("state")))) {
                    throw new IOException("fetching " + r.repo() + " on the Vast.ai machine failed"
                            + (d.get().get("error") != null ? ": " + d.get().get("error") : ""));
                }
                waitOr(cancelled, deadline, "fetching " + r.repo() + " onto the Vast.ai machine");
            }
        }
        return HostProvider.PREFIX + host.id() + "/" + name;
    }

    private static boolean has(WorkerClient client, String name) throws IOException {
        for (ModelInfo m : client.models()) {
            if (m.id().equals(name)) {
                return true;
            }
        }
        return false;
    }

    private void waitOr(BooleanSupplier cancelled, long deadline, String doing) throws IOException, InterruptedException {
        if (cancelled.getAsBoolean()) {
            throw new IOException("stopped while " + doing + "; the machine stays rented for the next task");
        }
        if (System.currentTimeMillis() > deadline) {
            throw new IOException(doing + " took too long");
        }
        Thread.sleep(pollMillis);
    }

    // a machine this runner rented before that is still being set up or ready
    private static Optional<Map<String, Object>> find(VastRentals vr, String tag) {
        return vr.list().stream()
                .filter(m -> tag.equals(m.get("tag")) && Json.integer(m, "ended", 0) == 0)
                .filter(m -> !List.of("failed", "destroyed", "destroying").contains(String.valueOf(m.get("stage"))))
                .findFirst();
    }

    private Map<String, Object> rent(VastRentals vr, String tag, Runner r) throws IOException, InterruptedException {
        VastAi v = vast.get();
        if (v == null) {
            throw new IOException("Vast.ai is not set up in this studio: add its key in Settings");
        }
        List<Map<String, Object>> offers = v.offers(r.gpu(), r.minGpuRam(), 1, r.maxPrice());
        if (offers.isEmpty()) {
            throw new IOException("Vast.ai has no " + (r.gpu().isEmpty() ? "GPU" : r.gpu()) + " machine for at most $"
                    + r.maxPrice() + " an hour now");
        }
        VastOffer offer = VastAi.toOffer(offers.getFirst());
        LOG.log(System.Logger.Level.INFO, "renting " + offer.gpuName() + " at $" + offer.dollarsPerHour() + "/h for " + tag);
        return vr.rent(offer, offer.dollarsPerHour(), r.hours(), tag);
    }
}
