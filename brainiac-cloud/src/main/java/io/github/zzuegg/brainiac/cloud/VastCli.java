package io.github.zzuegg.brainiac.cloud;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Rents a GPU on Vast.ai, runs Brainiac on it and destroys it again.
 *
 * <pre>
 * ./gradlew :brainiac-cloud:run --args="offers"                            # cheapest suitable GPUs
 * ./gradlew :brainiac-cloud:run --args="offers --gpu 'RTX 4090' --max-price 0.5"
 * ./gradlew :brainiac-cloud:run --args="run --offer 123456"                # smoke test: Vulkan, tests, benchmarks
 * ./gradlew :brainiac-cloud:run --args="run --offer 123456 --job tests"    # the whole test suite
 * ./gradlew :brainiac-cloud:run --args="run --offer 123456 --job run -- ./gradlew :brainiac-examples:run --args=iris"
 * ./gradlew :brainiac-cloud:run --args="worker --offer 123456"             # a studio worker behind an SSH tunnel
 * ./gradlew :brainiac-cloud:run --args="instances"                         # what is rented now
 * ./gradlew :brainiac-cloud:run --args="destroy 7654321"                   # or: destroy --all
 * </pre>
 *
 * <p>The API key comes from {@code VAST_API_KEY} or the studio's key file (see {@link VastKey})
 * and is never printed. Renting asks first, showing the offer and its price, unless
 * {@code --yes} is given. The instance is destroyed when the job ends, fails, runs past
 * {@code --max-minutes} or the launcher is stopped with Ctrl+C; {@code --keep} keeps it, and
 * {@code --instance ID} runs on one that is already rented.
 *
 * <p>A host that does not come up, or whose SSH login does not work within {@code --ssh-minutes}
 * (3 by default), is destroyed and remembered like one without a Vulkan driver; with
 * {@code --cheapest} the next offer is rented. The instance fetches the commit from GitHub when
 * it is pushed there, else this checkout is uploaded ({@code --upload} always uploads).
 * {@code --secret NAME} passes a secret such as {@code HF_TOKEN} to the job; see {@link VastSecrets}.
 */
public final class VastCli {

    private VastCli() {
    }

    /** The parsed command line. */
    private static final class Options {
        String command;
        long offer = -1;
        long instance = -1;
        boolean cheapest;
        String job = "smoke";
        List<String> jobArgs = new ArrayList<>();
        VastQuery query = VastQuery.defaults();
        VastLaunch launch = VastLaunch.defaults();
        boolean yes;
        boolean keep;
        boolean all;
        int maxMinutes = 120;
        int localPort = 7870;
        Path identity;
        List<Long> ids = new ArrayList<>();
        List<Long> skipped = new ArrayList<>();
        int minDriver;
        VastBadHosts bad;
        int sshMinutes = 3;
        boolean upload;
        List<String> secretSpecs = new ArrayList<>();
        VastSecrets secrets = VastSecrets.NONE;
    }

    /** The driver's major version, such as 595 for 595.91.07; 0 when unknown. */
    static int driverMajor(VastOffer f) {
        String v = f.driverVersion();
        int dot = v.indexOf('.');
        try {
            return Integer.parseInt(dot < 0 ? v.strip() : v.substring(0, dot).strip());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /**
     * Runs a command.
     *
     * @param args the command and its options; {@code help} lists them
     * @throws Exception if it fails
     */
    public static void main(String[] args) throws Exception {
        Options o = parse(args);
        if (o.command.equals("help")) {
            usage();
            return;
        }
        String key = VastKey.find().orElse(null);
        if (key == null) {
            System.err.println("No Vast.ai API key. Set VAST_API_KEY, or save it in the studio (Training -> Vast.ai),"
                    + " which keeps it in ~/.brainiac/studio/keys.properties.");
            System.exit(1);
        }
        VastClient vast = new VastClient(key);
        switch (o.command) {
            case "offers" -> offers(vast, o);
            case "instances" -> {
                List<VastInstance> list = vast.instances();
                System.out.println(list.isEmpty() ? "No instances: nothing is rented." : list.size() + " instance(s), billing until destroyed:");
                list.forEach(System.out::println);
            }
            case "destroy" -> destroy(vast, o);
            case "run", "worker" -> System.exit(rentAndRun(vast, o));
            default -> {
                usage();
                System.exit(1);
            }
        }
    }

    private static void usage() {
        System.out.println("""
                usage: vast offers [--gpu NAME,NAME] [--max-price DOLLARS] [--gpus N] [--min-gpu-ram GB] [--min-driver MAJOR] [--limit N]
                       vast run (--offer ID | --cheapest | --instance ID) [--job check|smoke|tests|bench|run]
                                [--image IMAGE] [--disk GB] [--max-minutes N] [--keep] [--yes] [--identity KEY]
                                [--ssh-minutes N] [--upload] [--secret NAME[=PROVIDER]]...
                                [-- ARGS for the job, e.g. the command for --job run]
                       vast worker (--offer ID | --cheapest | --instance ID) [--local-port 7870]
                                [--models DIR] [--keep] [--yes] [--ssh-minutes N] [--upload] [--secret NAME[=PROVIDER]]...

                --secret HF_TOKEN   sends this machine's Hugging Face token (environment, studio key or
                                    huggingface-cli login) to the job over SSH; never to disk or a log
                       vast instances
                       vast destroy ID... | --all""");
    }

    private static Options parse(String[] args) {
        Options o = new Options();
        o.command = args.length == 0 ? "help" : args[0];
        for (int i = 1; i < args.length; i++) {
            String a = args[i];
            switch (a) {
                case "--offer" -> o.offer = Long.parseLong(args[++i]);
                case "--instance" -> o.instance = Long.parseLong(args[++i]);
                case "--cheapest" -> o.cheapest = true;
                case "--job" -> o.job = args[++i];
                case "--gpu" -> o.query = o.query.withGpus(List.of(args[++i].split("\\s*,\\s*")));
                case "--min-driver" -> o.minDriver = Integer.parseInt(args[++i]);
                case "--gpus" -> o.query = o.query.withGpuCount(Integer.parseInt(args[++i]));
                case "--min-gpu-ram" -> o.query = o.query.withMinGpuMemory(Double.parseDouble(args[++i]));
                case "--max-price" -> o.query = o.query.withMaxPrice(Double.parseDouble(args[++i]));
                case "--limit" -> o.query = o.query.withLimit(Integer.parseInt(args[++i]));
                case "--image" -> o.launch = o.launch.withImage(args[++i]);
                case "--disk" -> o.launch = o.launch.withDisk(Integer.parseInt(args[++i]));
                case "--max-minutes" -> o.maxMinutes = Integer.parseInt(args[++i]);
                case "--local-port" -> o.localPort = Integer.parseInt(args[++i]);
                case "--models" -> o.jobArgs.addAll(List.of("--models", args[++i]));
                case "--identity" -> o.identity = Path.of(args[++i]);
                case "--ssh-minutes" -> o.sshMinutes = Integer.parseInt(args[++i]);
                case "--upload" -> o.upload = true;
                case "--secret" -> o.secretSpecs.add(args[++i]);
                case "--yes" -> o.yes = true;
                case "--keep" -> o.keep = true;
                case "--all" -> o.all = true;
                case "--" -> {
                    o.jobArgs.addAll(List.of(args).subList(i + 1, args.length));
                    i = args.length;
                }
                default -> {
                    if (o.command.equals("destroy") && a.matches("\\d+")) {
                        o.ids.add(Long.parseLong(a));
                    } else {
                        throw new IllegalArgumentException("unknown option " + a + "; try: vast help");
                    }
                }
            }
        }
        return o;
    }

    private static void offers(VastClient vast, Options o) throws IOException {
        List<VastOffer> list = new ArrayList<>(vast.offers(o.query));
        list.removeIf(f -> driverMajor(f) < o.minDriver);
        VastBadHosts bad = VastBadHosts.load();
        if (list.isEmpty()) {
            System.out.println("No offers match. Try --max-price higher or other --gpu names.");
        }
        System.out.printf(Locale.ROOT, "%-10s %-14s %6s %9s %-12s %7s %6s  %s%n",
                "offer", "gpu", "GB", "$/hour", "driver", "reliab", "Mbps", "location");
        for (VastOffer f : list) {
            System.out.printf(Locale.ROOT, "%-10d %-14s %6.0f %9.3f %-12s %7.3f %6.0f  %s%n", f.id(),
                    (f.gpus() > 1 ? f.gpus() + "x " : "") + f.gpuName(), f.gpuMemoryGb(), f.dollarsPerHour(),
                    f.driverVersion(), f.reliability(), f.downloadMbps(), f.location() + (bad.contains(f) ? "  (failed the Vulkan check before)" : ""));
        }
    }

    private static void destroy(VastClient vast, Options o) throws IOException {
        List<Long> ids = new ArrayList<>(o.ids);
        if (o.all) {
            vast.instances().forEach(i -> ids.add(i.id()));
        }
        if (ids.isEmpty()) {
            System.out.println("Nothing to destroy.");
        }
        for (long id : ids) {
            vast.destroy(id);
            System.out.println("destroyed instance " + id);
        }
    }

    /** How many hosts --cheapest tries when the first ones have no usable Vulkan driver. */
    private static final int MAX_HOSTS = 3;

    private static int rentAndRun(VastClient vast, Options o) throws Exception {
        try {
            // before anything is rented: a missing secret should not cost money
            o.secrets = VastSecrets.resolve(o.secretSpecs);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            return 1;
        }
        if (!o.secrets.isEmpty()) {
            System.out.println("Secrets for the job (values hidden): " + String.join(", ", o.secrets.names()));
        }
        if (o.instance >= 0) {
            o.keep = true;
            return runOn(vast, o, o.instance, vast.instance(o.instance).dollarsPerHour(), null, new String[1]);
        }
        VastBadHosts bad = VastBadHosts.load();
        o.bad = bad;
        for (int host = 1; ; host++) {
            VastOffer offer = pick(vast, o);
            if (offer == null) {
                return 1;
            }
            if (bad.contains(offer)) {
                System.out.println("Note: machine " + offer.machineId() + " failed the Vulkan check before (" + VastBadHosts.DEFAULT_FILE + ").");
            }
            String what = isUnlisted(offer) ? "offer #" + offer.id() + " (not in the current search listing; its price is shown once rented)"
                    : offer.toString();
            if (!o.yes && !confirm("Rent " + what + "?\nBilling runs until the instance is destroyed. [y/N] ")) {
                System.out.println("Nothing rented.");
                return 1;
            }
            System.out.println("Renting " + what);
            long id = -1;
            for (int tries = 0; ; tries++) {
                try {
                    id = vast.rent(offer.id(), o.launch);
                    break;
                } catch (IOException e) {
                    // an offer can stay listed after it is gone: with --cheapest, take the next one
                    // (each still within --max-price and the other limits the user gave)
                    if (!o.cheapest && e.getMessage().contains("no_such_ask")) {
                        System.out.println("Offer " + offer.id() + " is gone (Vast: no_such_ask); run 'offers' again or use --cheapest.");
                        return 1;
                    }
                    if (!o.cheapest || tries >= 4 || !e.getMessage().contains("no_such_ask")) {
                        throw e;
                    }
                    System.out.println("  offer " + offer.id() + " is no longer available; trying the next cheapest");
                    o.skipped.add(offer.id());
                    offer = pick(vast, o);
                    if (offer == null) {
                        return 1;
                    }
                    System.out.println("Renting " + offer);
                }
            }
            System.out.println("Rented instance " + id + (o.keep ? " (--keep: destroy it yourself when done)" : ""));
            double price = offer.dollarsPerHour();
            if (isUnlisted(offer)) {
                try {
                    VastInstance rented = vast.instance(id);
                    price = rented.dollarsPerHour();
                    System.out.printf(Locale.ROOT, "  instance %d: %s at $%.3f/h%n", id, rented.gpuName(), price);
                } catch (IOException e) {
                    System.out.println("  could not read the instance's price yet: " + e.getMessage());
                }
            }
            String[] reason = new String[1];
            int code = runOn(vast, o, id, price, offer, reason);
            if (code != BAD_HOST) {
                return code;
            }
            // the instance is destroyed by now (unless --keep)
            bad.add(offer, reason[0]);
            System.out.println("Machine " + (offer.machineId() == 0 ? "of offer " + offer.id() : offer.machineId()) + ": " + reason[0]
                    + (offer.machineId() == 0 ? "." : "; it is skipped from now on (" + VastBadHosts.DEFAULT_FILE + ")."));
            if (!o.cheapest || host >= MAX_HOSTS) {
                System.out.println(o.cheapest ? "Gave up after " + host + " hosts." : "Pick another offer, or use --cheapest.");
                return 2;
            }
            o.skipped.add(offer.id());
            System.out.println("Trying the next cheapest offer.");
        }
    }

    /**
     * {@link #runOn}'s result when the host is unusable: it failed {@code setup.sh check}, never
     * came up, or its SSH login did not work in time.
     */
    private static final int BAD_HOST = -2;

    /** An offer given with {@code --offer} that the search did not list: only its id is known. */
    private static boolean isUnlisted(VastOffer f) {
        return Double.isNaN(f.dollarsPerHour());
    }

    /**
     * Runs the job on a rented instance and destroys it afterwards, unless {@code --keep}.
     *
     * @param reason gets why, when the result is {@link #BAD_HOST}
     * @return the job's exit code, or {@link #BAD_HOST} when the host is unusable
     */
    private static int runOn(VastClient vast, Options o, long id, double price, VastOffer offer, String[] reason) throws Exception {
        boolean worker = o.command.equals("worker");
        long started = System.nanoTime();
        AtomicBoolean destroyed = new AtomicBoolean(o.keep);
        Runnable cleanup = () -> {
            if (destroyed.compareAndSet(false, true)) {
                try {
                    vast.destroy(id);
                    double hours = (System.nanoTime() - started) / 3.6e12;
                    System.out.printf(Locale.ROOT, "Destroyed instance %d after %.1f minutes, about $%.3f.%n",
                            id, hours * 60, hours * price);
                } catch (IOException e) {
                    System.err.println("COULD NOT DESTROY INSTANCE " + id + ": " + e.getMessage()
                            + "\nDestroy it at https://cloud.vast.ai/instances/ or with: vast destroy " + id);
                }
            }
        };
        // Ctrl+C, closing the window
        Thread hook = new Thread(cleanup, "vast-destroy");
        Runtime.getRuntime().addShutdownHook(hook);
        // --max-minutes: a run that hangs still ends
        Thread watchdog = Thread.ofVirtual().start(() -> {
            try {
                Thread.sleep(Duration.ofMinutes(o.maxMinutes));
                if (!worker) {
                    System.err.println("Stopping: the run took longer than --max-minutes " + o.maxMinutes);
                    cleanup.run();
                    Runtime.getRuntime().halt(3);
                }
            } catch (InterruptedException e) {
                // the run ended first
            }
        });
        try {
            VastInstance ready;
            VastRunner runner = new VastRunner(Path.of("").toAbsolutePath(), o.identity);
            try {
                ready = vast.awaitReady(id, Duration.ofMinutes(20));
                System.out.println("Instance " + id + " is running; waiting for SSH at " + ready.sshHost() + ":" + ready.sshPort()
                        + " (up to " + o.sshMinutes + " minutes)");
                runner.awaitSsh(ready, Duration.ofMinutes(o.sshMinutes));
            } catch (IOException e) {
                // a host that never starts, rejects the key or whose SSH proxy is dead: billing for
                // nothing, so it is given up on (and destroyed in finally) instead of waited for
                System.out.println(e.getMessage());
                if (offer == null) {
                    return 1;
                }
                reason[0] = e.getMessage().contains("over SSH") ? "SSH login failed within " + o.sshMinutes + " minutes"
                        : "the instance did not start";
                // an unusable host is destroyed even with --keep
                destroyed.set(false);
                return BAD_HOST;
            }
            String stamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"));
            Path dir = Path.of("build", "vast");
            Files.createDirectories(dir);
            System.out.printf(Locale.ROOT, "SSH works after %.1f minutes. Checking the GPU's Vulkan driver%n",
                    (System.nanoTime() - started) / 6e10);
            int check = runner.check(ready, dir.resolve("instance-" + id + "-check-" + stamp + ".log"));
            if (check != 0) {
                System.out.println();
                System.out.println(check == 2 ? "This machine cannot run Brainiac (see BRAINIAC_SETUP above)."
                        : "The Vulkan check failed (exit " + check + ").");
                reason[0] = "setup.sh check failed: no usable NVIDIA Vulkan driver";
                if (check == 2 && offer != null) {
                    destroyed.set(false);
                    return BAD_HOST;
                }
                return check;
            }
            if (!worker && o.job.equals("check")) {
                return 0;
            }
            System.out.printf(Locale.ROOT, "Vulkan works after %.1f minutes. Copying this checkout (HEAD) to the instance%n",
                    (System.nanoTime() - started) / 6e10);
            runner.ship(ready, !o.upload);
            System.out.printf(Locale.ROOT, "Copied after %.1f minutes; starting job %s%n", (System.nanoTime() - started) / 6e10,
                    worker ? "worker" : o.job);
            if (worker) {
                return worker(runner, ready, o);
            }
            Path log = dir.resolve("instance-" + id + "-" + o.job + "-" + stamp + ".log");
            int code = runner.run(ready, o.job, o.jobArgs, log, o.secrets);
            System.out.println();
            System.out.println(code == 0 ? "Job " + o.job + " finished." : code == 2
                    ? "This machine cannot run Brainiac (see BRAINIAC_SETUP above)." : "Job " + o.job + " failed (exit " + code + ").");
            System.out.println("Log: " + log.toAbsolutePath());
            return code;
        } finally {
            watchdog.interrupt();
            cleanup.run();
            try {
                Runtime.getRuntime().removeShutdownHook(hook);
            } catch (IllegalStateException e) {
                // already shutting down
            }
        }
    }

    private static int worker(VastRunner runner, VastInstance ready, Options o) throws Exception {
        byte[] bytes = new byte[32];
        new SecureRandom().nextBytes(bytes);
        String token = HexFormat.of().formatHex(bytes);
        Process p = runner.startWorker(ready, token, o.localPort, o.jobArgs, o.secrets);
        // Ctrl+C: stop the worker on the instance too (a kept instance would keep it running)
        Thread stop = new Thread(() -> {
            p.destroy();
            try {
                runner.stopWorker(ready);
            } catch (IOException | InterruptedException e) {
                System.err.println("could not stop the worker on instance " + ready.id() + ": " + e.getMessage());
            }
        }, "vast-stop-worker");
        Runtime.getRuntime().addShutdownHook(stop);
        try (BufferedReader out = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = out.readLine()) != null) {
                System.out.println(o.secrets.mask(line));
                if (line.startsWith("BRAINIAC_WORKER ready")) {
                    System.out.println();
                    System.out.println("The worker is reachable through the SSH tunnel. In the studio, Training -> Add a host:");
                    System.out.println("  name:  Vast " + ready.gpuName() + " #" + ready.id());
                    System.out.println("  url:   127.0.0.1:" + o.localPort);
                    System.out.println("  token: " + token);
                    System.out.println("Press Ctrl+C to stop" + (o.keep ? "." : " and destroy the instance."));
                }
            }
        }
        int code = p.waitFor();
        try {
            Runtime.getRuntime().removeShutdownHook(stop);
        } catch (IllegalStateException e) {
            // already shutting down: the hook stops the worker
            return code;
        }
        runner.stopWorker(ready);
        return code;
    }

    private static VastOffer pick(VastClient vast, Options o) throws IOException {
        if (o.offer >= 0) {
            // look the offer up again for its current price; Vast returns a different subset of
            // the matching offers on each search, so a live offer can be missing from one listing
            for (int search = 0; search < 3; search++) {
                for (VastOffer f : vast.offers(o.query.withLimit(1000).withMaxPrice(0))) {
                    if (f.id() == o.offer) {
                        return f;
                    }
                }
            }
            // not listed is not gone: renting it tells (Vast answers no_such_ask for a gone offer)
            return unlisted(o.offer);
        }
        if (o.cheapest) {
            List<VastOffer> list = new ArrayList<>(vast.offers(o.query));
            list.removeIf(f -> o.skipped.contains(f.id()) || driverMajor(f) < o.minDriver || (o.bad != null && o.bad.contains(f)));
            if (list.isEmpty()) {
                System.out.println("No offers match.");
                return null;
            }
            return list.getFirst();
        }
        System.out.println("Give --offer ID (from 'offers'), --cheapest, or --instance ID.");
        return null;
    }

    /** An offer known only by its id: price unknown (NaN), machine unknown (0). */
    static VastOffer unlisted(long id) {
        return new VastOffer(id, "?", 1, 0, Double.NaN, 0, "?", "?", 0, 0, 0, false, 0);
    }

    private static boolean confirm(String question) throws IOException {
        System.out.print(question);
        System.out.flush();
        String answer = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)).readLine();
        return answer != null && answer.strip().toLowerCase(Locale.ROOT).startsWith("y");
    }
}
