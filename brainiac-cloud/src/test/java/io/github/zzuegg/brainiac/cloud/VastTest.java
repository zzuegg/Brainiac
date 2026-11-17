package io.github.zzuegg.brainiac.cloud;

import io.github.zzuegg.brainiac.io.Json;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VastTest {

    @Test
    void queryAsksForOneVerifiedGpuCheapestFirst() {
        String q = Json.write(VastQuery.defaults().withGpus(List.of("RTX 4090")).withMaxPrice(0.5).toJson());
        assertTrue(q.contains("\"gpu_name\":{\"in\":[\"RTX 4090\"]}"), q);
        assertTrue(q.contains("\"num_gpus\":{\"eq\":1}"), q);
        assertTrue(q.contains("\"dph_total\":{\"lte\":0.5}"), q);
        assertTrue(q.contains("\"order\":[[\"dph_total\",\"asc\"]]"), q);
        assertTrue(q.contains("\"verified\":{\"eq\":true}"), q);
    }

    @Test
    void launchAsksForVulkanAndSsh() {
        Map<String, Object> b = VastLaunch.defaults().exposePorts(7870).toJson();
        assertEquals("ssh_proxy", b.get("runtype"));
        Map<String, Object> env = Json.object(b.get("env"));
        assertEquals("all", env.get("NVIDIA_DRIVER_CAPABILITIES"));
        assertEquals("1", env.get("-p 7870:7870"));
    }

    @Test
    void remembersBadHosts(@org.junit.jupiter.api.io.TempDir java.nio.file.Path dir) {
        java.nio.file.Path file = dir.resolve("vast/bad-machines.txt");
        VastOffer f = new VastOffer(1, "RTX 4090", 1, 24, 0.3, 0.99, "580.142", "Jiangsu, CN", 500, 100, 16, true, 4711);
        VastOffer other = new VastOffer(2, "RTX 4090", 1, 24, 0.3, 0.99, "580.142", "Jiangsu, CN", 500, 100, 16, true, 4712);
        VastBadHosts bad = new VastBadHosts(file);
        assertFalse(bad.contains(f));
        bad.add(f, "no usable NVIDIA Vulkan driver");
        bad.add(f, "again");
        assertTrue(bad.contains(f));
        VastBadHosts reread = new VastBadHosts(file);
        assertTrue(reread.contains(f));
        assertFalse(reread.contains(other));
        assertFalse(reread.contains(new VastOffer(3, "RTX 4090", 1, 24, 0.3, 0.99, "?", "?", 0, 0, 0, true, 0)));
    }

    @Test
    void readsOffersAndInstances() {
        VastOffer f = VastOffer.from(Json.object(Json.parse("""
                {"id": 123, "gpu_name": "RTX 4090", "num_gpus": 1, "gpu_ram": 24564, "dph_total": 0.3567,
                 "reliability2": 0.995, "driver_version": "560.35.03", "geolocation": "Norway, NO",
                 "inet_down": 812.5, "disk_space": 120.0, "verification": "verified", "machine_id": 4711}""")));
        assertEquals(123, f.id());
        assertEquals(4711, f.machineId());
        assertEquals(24, Math.round(f.gpuMemoryGb()));
        assertEquals(0.3567, f.dollarsPerHour());
        assertTrue(f.verified());

        VastInstance i = VastInstance.from(Json.object(Json.parse("""
                {"id": 99, "actual_status": "running", "status_msg": "", "gpu_name": "RTX 4090", "dph_total": 0.36,
                 "ssh_host": "ssh5.vast.ai", "ssh_port": 12345, "public_ipaddr": "1.2.3.4\\n",
                 "ports": {"7870/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40001"}], "22/tcp": null}}""")));
        assertTrue(i.ready());
        assertEquals(Optional.of("http://1.2.3.4:40001"), i.url(7870));
        assertEquals(Optional.empty(), i.url(22));

        VastInstance loading = VastInstance.from(Json.object(Json.parse("""
                {"id": 99, "actual_status": null, "ssh_host": null}""")));
        assertFalse(loading.ready());
    }

    @Test
    void findsTheKeyInTheEnvironmentFirstThenTheStudiosFile(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("keys.properties");
        assertEquals(Optional.empty(), VastKey.find(Map.of(), file));
        Files.writeString(file, "openai.key.default=sk-x\nvastai.order=default\nvastai.key.default=abc123\n");
        assertEquals(Optional.of("abc123"), VastKey.find(Map.of(), file));
        assertEquals(Optional.of("fromenv"), VastKey.find(Map.of("VAST_API_KEY", " fromenv "), file));
        Files.writeString(file, "vast.key.work=old\n");
        assertEquals(Optional.of("old"), VastKey.find(Map.of(), file));
    }

    @Test
    void quotesRemoteWords() {
        assertEquals("./gradlew", VastRunner.quote("./gradlew"));
        assertEquals("'--args=bench mnist'", VastRunner.quote("--args=bench mnist"));
        assertEquals("'it'\\''s'", VastRunner.quote("it's"));
    }
}
