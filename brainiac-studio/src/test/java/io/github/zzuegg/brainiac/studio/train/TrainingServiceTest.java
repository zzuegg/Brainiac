package io.github.zzuegg.brainiac.studio.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.studio.builder.LayerCatalog;
import io.github.zzuegg.brainiac.studio.builder.ModelGraph;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Runs that train a built model on built-in data, with options found in the installed modules. */
class TrainingServiceTest {

    private static final String ADAM = "io.github.zzuegg.brainiac.train.Optimizer#adam(float)";
    private static final String CROSS_ENTROPY = "io.github.zzuegg.brainiac.nn.Loss#CROSS_ENTROPY";
    private static final String ACCURACY = "io.github.zzuegg.brainiac.train.Metric#ACCURACY";
    private static final String DENSE = "io.github.zzuegg.brainiac.nn.Dense#spec(int,Activation,Init)";

    @TempDir
    Path folder;

    private void buildModel(String name, int inputs, int outputs) throws Exception {
        Map<String, Object> graph = Json.object(Json.parse("""
                {"nodes": [
                  {"id": "in", "op": "input", "shape": [%d]},
                  {"id": "h", "op": "layer", "entry": "%s", "args": [32, "RELU", "HE"], "inputs": ["in"]},
                  {"id": "out", "op": "layer", "entry": "%s", "args": [%d, "SOFTMAX", "XAVIER"], "inputs": ["h"]}
                ]}""".formatted(inputs, DENSE, DENSE, outputs)));
        Model m = new ModelGraph(LayerCatalog.discover()).build(graph, 3);
        Files.createDirectories(folder.resolve("models"));
        m.save(folder.resolve("models").resolve(name + ".brainiac"));
    }

    private static Map<String, Object> spec(String model, int epochs) {
        return Json.object(Json.parse("""
                {"model": "%s", "data": {"kind": "builtin", "name": "spirals"},
                 "optimizer": {"entry": "%s", "args": [0.02]}, "loss": {"entry": "%s"},
                 "metrics": ["%s"], "epochs": %d, "batchSize": 32, "validation": 0.2, "seed": 5, "device": "cpu"}
                """.formatted(model, ADAM, CROSS_ENTROPY, ACCURACY, epochs)));
    }

    private static TrainingRun await(TrainingRun run) throws InterruptedException {
        for (int i = 0; i < 1200 && run.state().active(); i++) {
            Thread.sleep(50);
        }
        return run;
    }

    @Test
    void theOptionsAreFoundInTheInstalledModules() {
        try (TrainingService t = new TrainingService(folder)) {
            Map<String, Object> o = t.optionsJson();
            List<String> optimizers = keys(o.get("optimizers"));
            assertTrue(optimizers.contains(ADAM), optimizers.toString());
            assertTrue(optimizers.stream().anyMatch(k -> k.contains("#muon(")), optimizers.toString());
            assertTrue(optimizers.stream().anyMatch(k -> k.contains("#lion(")), optimizers.toString());
            assertTrue(keys(o.get("losses")).contains(CROSS_ENTROPY), o.get("losses").toString());
            assertTrue(keys(o.get("metrics")).contains(ACCURACY));
            assertTrue(keys(o.get("schedules")).stream().anyMatch(k -> k.contains("#cosine(")));
            assertFalse(Json.array(o.get("devices")).isEmpty());
            Json.write(o);
        }
    }

    @Test
    void aSavedModelAnswersASampleOfTheDataAndValuesTypedIn() throws Exception {
        buildModel("tiny", 2, 3);
        try (TrainingService t = new TrainingService(folder)) {
            Map<String, Object> sample = t.predict(Json.object(Json.parse(
                    "{\"model\": \"tiny\", \"data\": {\"kind\": \"builtin\", \"name\": \"spirals\"}, \"index\": 7}")));
            assertEquals(7, sample.get("index"));
            assertEquals(900, sample.get("samples"));
            assertEquals(3, Json.array(sample.get("output")).size());
            int best = ((Number) sample.get("best")).intValue();
            assertTrue(best >= 0 && best < 3, sample.toString());
            int want = ((Number) sample.get("expectedBest")).intValue();
            assertTrue(want >= 0 && want < 3, sample.toString());
            Json.write(sample);

            Map<String, Object> own = t.predict(Json.object(Json.parse("{\"model\": \"tiny\", \"values\": [0.5, -0.25], \"full\": true}")));
            assertEquals(List.of(0.5, -0.25), own.get("input"));
            assertEquals(2, Json.array(own.get("input")).size());
            assertFalse(own.containsKey("expected"));
            double sum = Json.array(own.get("output")).stream().mapToDouble(v -> ((Number) v).doubleValue()).sum();
            assertEquals(1.0, sum, 1e-4, "a softmax output sums to one");

            IllegalArgumentException wrong = assertThrows(IllegalArgumentException.class,
                    () -> t.predict(Json.object(Json.parse("{\"model\": \"tiny\", \"values\": [1, 2, 3]}"))));
            assertTrue(wrong.getMessage().contains("takes 2 values"), wrong.getMessage());
            assertThrows(IllegalArgumentException.class,
                    () -> t.predict(Json.object(Json.parse("{\"model\": \"missing\", \"values\": [1, 2]}"))));
        }
    }

    private static List<String> keys(Object list) {
        return Json.array(list).stream().map(e -> String.valueOf(Json.object(e).get("key"))).toList();
    }

    @Test
    void aRunTrainsSavesTheResultAndIsKeptAfterARestart() throws Exception {
        buildModel("tiny", 2, 3);
        String id;
        try (TrainingService t = new TrainingService(folder)) {
            Map<String, Object> check = t.check(spec("tiny", 30));
            assertEquals(true, check.get("ok"), check.toString());
            assertEquals(900, ((Number) check.get("samples")).intValue());
            TrainingRun run = await(t.start(spec("tiny", 30)));
            Map<String, Object> json = run.toJson(true);
            assertEquals("finished", json.get("state"), json.toString());
            List<Object> history = Json.array(json.get("history"));
            assertEquals(30, history.size());
            float first = ((Number) Json.object(history.getFirst()).get("loss")).floatValue();
            float last = ((Number) Json.object(history.getLast()).get("loss")).floatValue();
            assertTrue(last < first * 0.8f, first + " -> " + last);
            Map<String, Object> val = Json.object(Json.object(history.getLast()).get("validationMetrics"));
            assertTrue(((Number) val.get("accuracy")).floatValue() > 0.5f, val.toString());
            assertFalse(Json.array(json.get("curve")).isEmpty());
            assertEquals("tiny-run" + run.id(), json.get("output"));
            assertTrue(Files.exists(folder.resolve("models/tiny-run" + run.id() + ".brainiac")));
            id = run.id();
        }
        try (TrainingService again = new TrainingService(folder)) {
            TrainingRun kept = again.runs().getFirst();
            assertEquals(id, kept.id());
            assertEquals(TrainingRun.State.FINISHED, kept.state());
            assertEquals(30, Json.array(kept.toJson(false).get("history")).size());
            TrainingRun next = again.start(spec("tiny", 1));
            assertEquals(String.valueOf(Integer.parseInt(id) + 1), next.id());
            await(next);
            assertTrue(again.delete(id));
            assertFalse(Files.exists(folder.resolve("runs/" + id + ".json")));
        }
    }

    @Test
    void aRunIsRenamedGoesToTheTrashAndComesBackWithoutItsIdBeingGivenAgain() throws Exception {
        buildModel("small", 2, 3);
        Path bin = Files.createDirectories(folder.resolve("bin"));
        try (TrainingService t = new TrainingService(folder)) {
            java.util.List<String> told = new java.util.ArrayList<>();
            t.onStart(r -> told.add(r.id()));
            TrainingRun run = await(t.start(spec("small", 1)));
            assertEquals(List.of(run.id()), told);
            t.rename(run.id(), "first try");
            assertEquals("first try", run.name());
            t.detach(run.id(), bin);
            assertTrue(t.run(run.id()).isEmpty());
            assertTrue(Files.exists(bin.resolve("run.json")));
        }
        try (TrainingService t = new TrainingService(folder)) {
            TrainingRun next = await(t.start(spec("small", 1)));
            assertEquals("2", next.id(), "the id of the run in the trash is not given again");
            t.attach("1", bin);
            assertEquals("first try", t.run("1").orElseThrow().name());
            assertThrows(IllegalArgumentException.class, () -> t.rename("1", " "));
        }
    }

    @Test
    void aRunCanBeStoppedAndKeepsWhatItLearned() throws Exception {
        buildModel("long", 2, 3);
        try (TrainingService t = new TrainingService(folder)) {
            TrainingRun run = t.start(spec("long", 100_000));
            for (int i = 0; i < 400 && run.toJson(false).get("step") instanceof Number n && n.longValue() < 5; i++) {
                Thread.sleep(25);
            }
            assertTrue(t.stop(run.id()));
            await(run);
            assertEquals(TrainingRun.State.STOPPED, run.state());
            assertNotNull(run.toJson(false).get("output"));
            assertThrows(IllegalArgumentException.class, () -> t.start(Map.of("model", "nope")));
        }
    }

    @Test
    void dataThatDoesNotFitTheModelIsExplained() throws Exception {
        buildModel("wide", 5, 3);
        try (TrainingService t = new TrainingService(folder)) {
            Map<String, Object> check = t.check(spec("wide", 1));
            assertEquals(false, check.get("ok"));
            assertTrue(String.valueOf(check.get("problem")).contains("each sample has 2 values"), check.toString());
            TrainingRun run = await(t.start(spec("wide", 1)));
            assertEquals(TrainingRun.State.FAILED, run.state());
            assertTrue(String.valueOf(run.toJson(false).get("error")).contains("each sample has 2 values"));
        }
    }

    @Test
    void theBuiltInSetsHaveTheShapesTheyPromise() {
        for (DataSources.Builtin b : DataSources.BUILTINS) {
            Dataset d = DataSources.builtin(b.name());
            int in = b.input().stream().reduce(1, (x, y) -> x * y);
            assertEquals(in, d.featureCount(), b.name());
            assertEquals(b.targets(), d.targetCount(), b.name());
        }
    }

    @Test
    void csvFilesInTheDataFolderAreListedAndRead() throws Exception {
        Path data = folder.resolve("datasets");
        Files.createDirectories(data);
        StringBuilder csv = new StringBuilder("a,b,kind\n");
        for (int i = 0; i < 20; i++) {
            csv.append(i).append(',').append(i * 2).append(',').append(i % 2 == 0 ? "even" : "odd").append('\n');
        }
        Files.writeString(data.resolve("numbers.csv"), csv);
        DataSources sources = new DataSources(data);
        Map<String, Object> list = sources.list();
        Map<String, Object> file = Json.object(Json.array(list.get("files")).getFirst());
        assertEquals("csv", file.get("kind"));
        assertEquals(List.of("a", "b", "kind"), file.get("columns"));
        Dataset d = sources.load(Map.of("kind", "csv", "file", "numbers.csv"));
        assertEquals(20, d.size());
        assertEquals(2, d.featureCount());
        assertEquals(2, d.targetCount());   // the last column as classes, one-hot
        assertThrows(IllegalArgumentException.class, () -> sources.load(Map.of("kind", "csv", "file", "missing.csv")));
    }
}
