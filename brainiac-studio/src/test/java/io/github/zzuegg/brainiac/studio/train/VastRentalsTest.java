package io.github.zzuegg.brainiac.studio.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.cloud.VastBadHosts;
import io.github.zzuegg.brainiac.cloud.VastInstance;
import io.github.zzuegg.brainiac.cloud.VastOffer;
import io.github.zzuegg.brainiac.studio.server.WorkerServer;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Machines rented on Vast.ai become hosts and are always destroyed again. */
class VastRentalsTest {

    @TempDir
    Path studio;

    @TempDir
    Path lab;

    /** Vast.ai and the machine, played here: the "machine" is a worker started in this process. */
    final class Fake implements VastRentals.Machines {
        final List<Long> rented = new CopyOnWriteArrayList<>();
        final List<Long> destroyed = new CopyOnWriteArrayList<>();
        volatile IOException failPrepare;
        volatile boolean slowRent;
        final AtomicBoolean releaseRent = new AtomicBoolean();
        final CountDownLatch asked = new CountDownLatch(1);
        final AtomicBoolean tunnelOpen = new AtomicBoolean();

        @Override
        public long rent(long offerId) {
            asked.countDown();
            // Vast.ai takes its time to answer
            while (slowRent && !releaseRent.get()) {
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    throw new AssertionError("renting was interrupted, so the instance would be lost");
                }
            }
            long id = 9000 + rented.size();
            rented.add(id);
            return id;
        }

        @Override
        public VastInstance awaitReady(long id, Duration timeout) {
            return new VastInstance(id, "running", "", "RTX 4090", 0.36, "ssh.example", 22, "", "", Map.of());
        }

        volatile boolean noVulkan;

        @Override
        public boolean usable(VastInstance i, Path log) {
            return !noVulkan;
        }

        @Override
        public void prepare(VastInstance i) throws IOException {
            if (failPrepare != null) {
                throw failPrepare;
            }
        }

        @Override
        public VastRentals.Tunnel startWorker(VastInstance i, String token, int localPort) throws IOException {
            WorkerServer w = WorkerServer.start(new InetSocketAddress(InetAddress.getLoopbackAddress(), localPort), lab,
                    new HostChatTest.Echo(), token);
            tunnelOpen.set(true);
            return new VastRentals.Tunnel() {
                @Override
                public boolean alive() {
                    return tunnelOpen.get();
                }

                @Override
                public InputStream output() {
                    return new ByteArrayInputStream("building\nworker listening\n".getBytes(StandardCharsets.UTF_8));
                }

                @Override
                public void close() {
                    tunnelOpen.set(false);
                    w.close();
                }
            };
        }

        @Override
        public void stopWorker(VastInstance i) {
        }

        @Override
        public void destroy(long id) {
            destroyed.add(id);
        }
    }

    private VastBadHosts bad() {
        return new VastBadHosts(studio.resolve("bad-machines.txt"));
    }

    private static VastOffer offer(long id, double price) {
        return new VastOffer(id, "RTX 4090", 1, 24, price, 0.99, "570.1", "Sweden, SE", 900, 100, 16, true, 77);
    }

    private static Map<String, Object> await(VastRentals v, Predicate<Map<String, Object>> done) throws InterruptedException {
        long end = System.nanoTime() + Duration.ofSeconds(20).toNanos();
        while (System.nanoTime() < end) {
            Map<String, Object> r = v.list().getFirst();
            if (done.test(r)) {
                return r;
            }
            Thread.sleep(50);
        }
        throw new AssertionError("timed out at " + v.list());
    }

    private static boolean stage(Map<String, Object> r, String s) {
        return s.equals(r.get("stage"));
    }

    @Test
    void aRentedMachineBecomesAHostAndDestroyingItRemovesTheHost() throws Exception {
        Fake f = new Fake();
        Hosts hosts = new Hosts(studio.resolve("hosts.properties"));
        try (VastRentals v = new VastRentals(hosts, f, studio.resolve("vast"), null, bad())) {
            Map<String, Object> started = v.rent(offer(1234, 0.36), 0.36, 2);
            assertEquals("renting", started.get("stage"));
            Map<String, Object> ready = await(v, r -> stage(r, "ready"));
            assertEquals(9000L, ((Number) ready.get("instance")).longValue());
            assertEquals(List.of(9000L), f.rented);
            Hosts.Host h = hosts.list().getFirst();
            assertEquals("Vast RTX 4090", h.name());
            assertEquals(h.id(), ready.get("host"));
            assertTrue(Files.readString(studio.resolve("vast/vast-9000.log")).contains("worker listening"),
                    "what the machine prints is kept");

            v.destroy(9000);
            Map<String, Object> gone = await(v, r -> stage(r, "destroyed"));
            assertEquals(List.of(9000L), f.destroyed);
            assertTrue(hosts.list().isEmpty(), "the host goes with the machine");
            assertFalse(f.tunnelOpen.get(), "the tunnel is closed");
            assertTrue(((Number) gone.get("ended")).longValue() > 0);
        }
    }

    @Test
    void nothingIsRentedForMoreThanTheUserAgreedTo() {
        Fake f = new Fake();
        try (VastRentals v = new VastRentals(new Hosts(studio.resolve("hosts.properties")), f, studio, null, bad())) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> v.rent(offer(1, 0.41), 0.36, 2));
            assertTrue(e.getMessage().contains("$0.410"), e.getMessage());
            assertThrows(IllegalArgumentException.class, () -> v.rent(offer(1, 0.36), 0.36, 0));
            assertThrows(IllegalArgumentException.class, () -> v.rent(offer(1, 0.36), 0.36, VastRentals.MAX_HOURS + 1));
            assertTrue(f.rented.isEmpty());
            assertTrue(v.list().isEmpty());
        }
        try (VastRentals v = new VastRentals(new Hosts(studio.resolve("hosts.properties")), f, studio, "no checkout", bad())) {
            assertEquals("no checkout", assertThrows(IllegalStateException.class,
                    () -> v.rent(offer(1, 0.36), 0.36, 2)).getMessage());
        }
    }

    @Test
    void aMachineThatCannotBeSetUpIsDestroyed() throws Exception {
        Fake f = new Fake();
        f.failPrepare = new IOException("ssh refused the key");
        try (VastRentals v = new VastRentals(new Hosts(studio.resolve("hosts.properties")), f, studio, null, bad())) {
            v.rent(offer(1, 0.36), 0.36, 2);
            Map<String, Object> r = await(v, x -> stage(x, "failed"));
            assertEquals(List.of(9000L), f.destroyed);
            assertTrue(((String) r.get("error")).contains("ssh refused the key"), r.toString());
            assertTrue(((String) r.get("error")).contains("destroyed"), r.toString());
        }
    }

    @Test
    void aMachineWithoutVulkanIsDestroyedAndSkippedFromThenOn() throws Exception {
        Fake f = new Fake();
        f.noVulkan = true;
        try (VastRentals v = new VastRentals(new Hosts(studio.resolve("hosts.properties")), f, studio, null, bad())) {
            v.rent(offer(1, 0.36), 0.36, 2);
            Map<String, Object> r = await(v, x -> stage(x, "failed"));
            assertEquals(List.of(9000L), f.destroyed);
            assertTrue(((String) r.get("error")).contains("Vulkan"), r.toString());
            assertTrue(bad().contains(offer(2, 0.30)), "another offer on the same machine is skipped");
        }
    }

    @Test
    void cancellingWhileVastAiRentsStillDestroysTheMachineItGives() throws Exception {
        Fake f = new Fake();
        f.slowRent = true;
        try (VastRentals v = new VastRentals(new Hosts(studio.resolve("hosts.properties")), f, studio, null, bad())) {
            String key = (String) v.rent(offer(1, 0.36), 0.36, 2).get("key");
            assertTrue(f.asked.await(5, TimeUnit.SECONDS));
            assertTrue(v.cancel(key));
            Thread.sleep(200);
            f.releaseRent.set(true);   // Vast.ai answers after all
            await(v, x -> stage(x, "destroyed"));
            assertEquals(List.of(9000L), f.destroyed, "the instance rented while cancelling is destroyed");
            assertFalse(v.cancel(key), "an ended rental cannot be cancelled again");
        }
    }

    @Test
    void theMachineIsDestroyedWhenItsHoursAreUpOrTheStudioCloses() throws Exception {
        Fake f = new Fake();
        Hosts hosts = new Hosts(studio.resolve("hosts.properties"));
        try (VastRentals v = new VastRentals(hosts, f, studio, null, bad(), Duration.ofMillis(100))) {
            v.rent(offer(1, 0.36), 0.36, 2.0 / 3600);   // two seconds
            await(v, x -> stage(x, "ready"));
            await(v, x -> stage(x, "destroyed"));
            assertEquals(List.of(9000L), f.destroyed);
            assertTrue(hosts.list().isEmpty());
        }
        VastRentals v = new VastRentals(hosts, f, studio, null, bad());
        v.rent(offer(1, 0.36), 0.36, 2);
        await(v, x -> stage(x, "ready"));
        v.close();
        assertEquals(List.of(9000L, 9001L), f.destroyed, "closing the studio destroys what it rented");
        assertTrue(hosts.list().isEmpty());
    }
}
