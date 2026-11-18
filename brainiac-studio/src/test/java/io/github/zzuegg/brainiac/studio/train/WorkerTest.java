package io.github.zzuegg.brainiac.studio.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog;
import io.github.zzuegg.brainiac.studio.builder.ModelGraph;
import io.github.zzuegg.brainiac.studio.server.WorkerServer;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** A studio sends runs to a worker on another machine (here, another folder on this one). */
class WorkerTest {

    private static final String DENSE = "io.github.zzuegg.brainiac.nn.Dense#spec(int,Activation,Init)";

    @TempDir
    Path studio;

    @TempDir
    Path lab;

    private void buildModel(String name, int inputs, int outputs) throws Exception {
        Map<String, Object> graph = Json.object(Json.parse("""
                {"nodes": [
                  {"id": "in", "op": "input", "shape": [%d]},
                  {"id": "h", "op": "layer", "entry": "%s", "args": [16, "RELU", "HE"], "inputs": ["in"]},
                  {"id": "out", "op": "layer", "entry": "%s", "args": [%d, "SOFTMAX", "XAVIER"], "inputs": ["h"]}
                ]}""".formatted(inputs, DENSE, DENSE, outputs)));
        Model m = new ModelGraph(LayerCatalog.discover()).build(graph, 3);
        Files.createDirectories(studio.resolve("models"));
        m.save(studio.resolve("models").resolve(name + ".brainiac"));
    }

    private static TrainingRun await(TrainingRun run) throws InterruptedException {
        for (int i = 0; i < 1200 && run.state().active(); i++) {
            Thread.sleep(50);
        }
        return run;
    }

    @Test
    void aRunTrainsOnTheWorkerAndTheModelComesBack() throws Exception {
        buildModel("net", 2, 2);
        Path data = studio.resolve("datasets");
        Files.createDirectories(data);
        StringBuilder csv = new StringBuilder("x,y,side\n");
        for (int i = 0; i < 200; i++) {
            double x = Math.sin(i * 1.7), y = Math.cos(i * 2.3);
            csv.append(x).append(',').append(y).append(',').append(x + y > 0 ? "up" : "down").append('\n');
        }
        Files.writeString(data.resolve("sides.csv"), csv);
        try (WorkerServer worker = WorkerServer.start(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), lab);
             TrainingService t = new TrainingService(studio)) {
            String url = "127.0.0.1:" + worker.port();
            assertThrows(IOException.class, () -> t.hosts().add("lab", url, "wrong-token-wrong-token-wrong-token"));
            Hosts.Host h = t.hosts().add("lab", url, worker.token());
            Map<String, Object> listed = t.hosts().toJson().getFirst();
            assertEquals(true, listed.get("online"));
            assertFalse(listed.toString().contains(worker.token()), "the token never goes to the page");
            Map<String, Object> spec = Json.object(Json.parse("""
                    {"model": "net", "host": "%s", "data": {"kind": "csv", "file": "sides.csv"},
                     "optimizer": {"entry": "io.github.zzuegg.brainiac.train.Optimizer#adam(float)", "args": [0.02]},
                     "loss": {"entry": "io.github.zzuegg.brainiac.nn.Loss#CROSS_ENTROPY"},
                     "metrics": ["io.github.zzuegg.brainiac.train.Metric#ACCURACY"],
                     "epochs": 15, "batchSize": 16, "validation": 0.2, "device": "cpu"}""".formatted(h.id())));
            TrainingRun run = await(t.start(spec));
            Map<String, Object> json = run.toJson(true);
            assertEquals("finished", json.get("state"), json.toString());
            assertEquals("lab", json.get("host"));
            assertEquals(15, Json.array(json.get("history")).size());
            assertFalse(Json.array(json.get("curve")).isEmpty());
            assertTrue(Files.exists(studio.resolve("models/net-run" + run.id() + ".brainiac")));
            Model back = Model.load(studio.resolve("models/net-run" + run.id() + ".brainiac"));
            assertEquals(2, back.outputShape().toArray()[0]);
            // the worker kept the file it was sent, under its hash, and sends the same one only once
            assertEquals(2, Files.list(lab.resolve("files")).count());   // the model and the CSV
            TrainingRun again = await(t.start(spec));
            assertEquals("finished", again.toJson(false).get("state"));
            assertEquals(2, Files.list(lab.resolve("files")).count());
            // hosts are kept with their token, in a file only this account can read
            assertTrue(Files.readString(studio.resolve("hosts.properties")).contains(worker.token()));
            assertTrue(t.hosts().remove(h.id()));
            assertTrue(t.hosts().list().isEmpty());
        }
    }

    @Test
    void aWorkerTrainsOnlyOnFilesItWasSent() throws Exception {
        try (WorkerServer worker = WorkerServer.start(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), lab)) {
            WorkerClient c = new WorkerClient("127.0.0.1:" + worker.port(), worker.token());
            Path model = Files.writeString(studio.resolve("x.brainiac"), "not really a model");
            String ref = c.upload(model);
            IOException e = assertThrows(IOException.class, () -> c.start(Map.of("data",
                    Map.of("kind", "csv", "file", "/etc/passwd")), ref));
            assertTrue(e.getMessage().contains("only trains on files sent to it"), e.getMessage());
        }
    }

    @Test
    void addressesAreReadTheWayPeopleTypeThem() {
        assertEquals("http://192.168.1.20:7870/", WorkerClient.normalize("192.168.1.20:7870").toString());
        assertEquals("http://lab:7870/", WorkerClient.normalize(" http://lab:7870 ").toString());
        assertThrows(IllegalArgumentException.class, () -> WorkerClient.normalize("ftp://lab"));
        assertThrows(IllegalArgumentException.class, () -> WorkerClient.normalize(""));
    }

    @Test
    void aWorkerListsItsModelDownloadsAndRefusesWhatIsNotARepository() throws Exception {
        try (WorkerServer worker = WorkerServer.start(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), lab)) {
            WorkerClient c = new WorkerClient("127.0.0.1:" + worker.port(), worker.token());
            assertEquals(java.util.List.of(), c.hubDownloads());
            IOException refused = assertThrows(IOException.class, () -> c.hubDownload("not a repository", "folder"));
            assertTrue(refused.getMessage().contains("not a Hugging Face repository"), refused.getMessage());
            assertFalse(c.hubCancel("1"), "there is no download to stop");
            WorkerClient stranger = new WorkerClient("127.0.0.1:" + worker.port(), "wrong-token-wrong-token-wrong-token");
            assertThrows(IOException.class, stranger::hubDownloads);
        }
    }
}
