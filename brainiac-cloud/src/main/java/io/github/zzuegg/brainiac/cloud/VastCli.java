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
                usage: vast offers [--gpu NAME,NAME] [--max-price DOLLARS] [--min-gpu-ram GB] [--limit N]
                       vast run (--offer ID | --cheapest | --instance ID) [--job check|smoke|tests|bench|run]
                                [--image IMAGE] [--disk GB] [--max-minutes N] [--keep] [--yes] [--identity KEY]
                                [-- ARGS for the job, e.g. the command for --job run]
                       vast worker (--offer ID | --cheapest | --instance ID) [--local-port 7870]
                                [--models DIR] [--keep] [--yes]
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
                case "--min-gpu-ram" -> o.query = o.query.withMinGpuMemory(Double.parseDouble(args[++i]));
                case "--max-price" -> o.query = o.query.withMaxPrice(Double.parseDouble(args[++i]));
                case "--limit" -> o.query = o.query.withLimit(Integer.parseInt(args[++i]));
                case "--image" -> o.launch = o.launch.withImage(args[++i]);
                case "--disk" -> o.launch = o.launch.withDisk(Integer.parseInt(args[++i]));
                case "--max-minutes" -> o.maxMinutes = Integer.parseInt(args[++i]);
                case "--local-port" -> o.localPort = Integer.parseInt(args[++i]);
                case "--models" -> o.jobArgs.addAll(List.of("--models", args[++i]));
                case "--identity" -> o.identity = Path.of(args[++i]);
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
        List<VastOffer> list = vast.offers(o.query);
        if (list.isEmpty()) {
            System.out.println("No offers match. Try --max-price higher or other --gpu names.");
        }
        System.out.printf(Locale.ROOT, "%-10s %-14s %6s %9s %-12s %7s %6s  %s%n",
                "offer", "gpu", "GB", "$/hour", "driver", "reliab", "Mbps", "location");
        for (VastOffer f : list) {
            System.out.printf(Locale.ROOT, "%-10d %-14s %6.0f %9.3f %-12s %7.3f %6.0f  %s%n", f.id(),
                    (f.gpus() > 1 ? f.gpus() + "x " : "") + f.gpuName(), f.gpuMemoryGb(), f.dollarsPerHour(),
                    f.driverVersion(), f.reliability(), f.downloadMbps(), f.location());
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

    private static int rentAndRun(VastClient vast, Options o) throws Exception {
        boolean worker = o.command.equals("worker");
        long id = o.instance;
        double price;
        if (id < 0) {
            VastOffer offer = pick(vast, o);
            if (offer == null) {
                return 1;
            }
            if (!o.yes && !confirm("Rent " + offer + "?\nBilling runs until the instance is destroyed. [y/N] ")) {
                System.out.println("Nothing rented.");
                return 1;
            }
            System.out.println("Renting " + offer);
            long rentedId = -1;
            for (int tries = 0; ; tries++) {
                try {
                    rentedId = vast.rent(offer.id(), o.launch);
                    break;
                } catch (IOException e) {
                    // an offer can stay listed after it is gone: with --cheapest, take the next one
                    // (each still within --max-price and the other limits the user gave)
                    if (!o.cheapest || tries >= 4 || !e.getMessage().contains("no_such_ask")) {
                        throw e;
                    }
                    System.out.println("  offer " + offer.id() + " is no longer available; trying the next cheapest");
                    long gone = offer.id();
                    o.skipped.add(gone);
                    offer = pick(vast, o);
                    if (offer == null) {
                        return 1;
                    }
                    System.out.println("Renting " + offer);
                }
            }
            id = rentedId;
            price = offer.dollarsPerHour();
            System.out.println("Rented instance " + id + (o.keep ? " (--keep: destroy it yourself when done)" : ""));
        } else {
            price = vast.instance(id).dollarsPerHour();
            o.keep = true;
        }
        long started = System.nanoTime();
        long rented = id;
        AtomicBoolean destroyed = new AtomicBoolean(o.keep);
        Runnable cleanup = () -> {
            if (destroyed.compareAndSet(false, true)) {
                try {
                    vast.destroy(rented);
                    double hours = (System.nanoTime() - started) / 3.6e12;
                    System.out.printf(Locale.ROOT, "Destroyed instance %d after %.1f minutes, about $%.3f.%n",
                            rented, hours * 60, hours * price);
                } catch (IOException e) {
                    System.err.println("COULD NOT DESTROY INSTANCE " + rented + ": " + e.getMessage()
                            + "\nDestroy it at https://cloud.vast.ai/instances/ or with: vast destroy " + rented);
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
            VastInstance ready = vast.awaitReady(id, Duration.ofMinutes(20));
            VastRunner runner = new VastRunner(Path.of("").toAbsolutePath(), o.identity);
            System.out.println("Instance " + id + " is running; waiting for SSH at " + ready.sshHost() + ":" + ready.sshPort());
            runner.awaitSsh(ready, Duration.ofMinutes(5));
            System.out.printf(Locale.ROOT, "SSH works after %.1f minutes. Copying this checkout (HEAD) to the instance%n",
                    (System.nanoTime() - started) / 6e10);
            runner.ship(ready);
            System.out.printf(Locale.ROOT, "Copied after %.1f minutes; starting job %s%n", (System.nanoTime() - started) / 6e10,
                    worker ? "worker" : o.job);
            if (worker) {
                return worker(runner, ready, o);
            }
            Path log = Path.of("build", "vast", "instance-" + id + "-" + o.job + "-"
                    + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")) + ".log");
            Files.createDirectories(log.getParent());
            int code = runner.run(ready, o.job, o.jobArgs, log);
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
        Process p = runner.startWorker(ready, token, o.localPort, o.jobArgs);
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
                System.out.println(line);
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
            // look the offer up again for its current price
            for (VastOffer f : vast.offers(o.query.withLimit(500).withMaxPrice(0))) {
                if (f.id() == o.offer) {
                    return f;
                }
            }
            System.out.println("Offer " + o.offer + " is gone or does not match the search; run 'offers' again.");
            return null;
        }
        if (o.cheapest) {
            List<VastOffer> list = new ArrayList<>(vast.offers(o.query));
            list.removeIf(f -> o.skipped.contains(f.id()));
            if (list.isEmpty()) {
                System.out.println("No offers match.");
                return null;
            }
            return list.getFirst();
        }
        System.out.println("Give --offer ID (from 'offers'), --cheapest, or --instance ID.");
        return null;
    }

    private static boolean confirm(String question) throws IOException {
        System.out.print(question);
        System.out.flush();
        String answer = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)).readLine();
        return answer != null && answer.strip().toLowerCase(Locale.ROOT).startsWith("y");
    }
}
