package io.github.zzuegg.brainiac.studio.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatListener;
import io.github.zzuegg.brainiac.studio.chat.ChatProvider;
import io.github.zzuegg.brainiac.studio.chat.ChatRequest;
import io.github.zzuegg.brainiac.studio.chat.ChatResult;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.Format;
import io.github.zzuegg.brainiac.studio.chat.Keys;
import io.github.zzuegg.brainiac.studio.chat.ModelInfo;
import io.github.zzuegg.brainiac.studio.train.TrainingService;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Agents think with any chat model, call the studio's tools, and wait for the user before acting. */
class AgentRunsTest {

    @TempDir
    Path data;

    private Script script;
    private ChatService chat;
    private TrainingService training;
    private AgentRuns agents;

    /** A model that says what the test tells it to, and remembers what it was sent. */
    static final class Script implements ChatProvider {
        final Deque<String> turns = new ArrayDeque<>();
        final List<ChatRequest> heard = new ArrayList<>();

        @Override
        public String id() {
            return "script";
        }

        @Override
        public String name() {
            return "Script";
        }

        @Override
        public boolean ready() {
            return true;
        }

        @Override
        public boolean local() {
            return true;
        }

        @Override
        public List<ModelInfo> models() {
            return List.of(new ModelInfo("script", "bot", "Bot", List.of("text"), 0, ""));
        }

        @Override
        public synchronized ChatResult chat(ChatRequest request, ChatListener listener) throws IOException {
            heard.add(request);
            String turn = turns.isEmpty() ? "I am out of words." : turns.pop();
            listener.text(turn);
            return new ChatResult("end", 10, 5);
        }
    }

    @BeforeEach
    void open() {
        script = new Script();
        chat = new ChatService(new Keys(data.resolve("keys.properties"), Map.of()), List.of(script),
                data.resolve("outputs"));
        training = new TrainingService(data);
        agents = new AgentRuns(chat, new StudioTools(chat, training), data.resolve("agents"));
    }

    @AfterEach
    void close() throws InterruptedException {
        // agents write their files from their own threads; let them end before the folder goes
        for (int i = 0; i < 200 && agents.list().stream().anyMatch(r -> "running".equals(r.get("state"))
                || "waiting".equals(r.get("state"))); i++) {
            agents.list().stream().filter(r -> "waiting".equals(r.get("state")))
                    .forEach(r -> agents.stop((String) r.get("id")));
            Thread.sleep(20);
        }
        agents.awaitIdle(5000);
        training.close();
    }

    private Map<String, Object> until(String id, Predicate<Map<String, Object>> done) throws InterruptedException {
        for (int i = 0; i < 200; i++) {
            Map<String, Object> r = agents.get(id).orElseThrow();
            if (done.test(r)) {
                return r;
            }
            Thread.sleep(20);
        }
        throw new AssertionError("the agent did not get there: " + agents.get(id).orElseThrow());
    }

    private static List<String> kinds(Map<String, Object> run) {
        return Json.array(run.get("steps")).stream().map(s -> (String) Json.object(s).get("kind")).toList();
    }

    @Test
    void aRunCanBeRenamedMovedAndTakenOutAndPutBack() throws Exception {
        script.turns.add("Done.");
        String id = (String) agents.start("Say done", "script/bot", "auto", 3, null).get("id");
        until(id, r -> !"running".equals(r.get("state")));

        agents.rename(id, "  My helper ");
        assertEquals("My helper", agents.get(id).orElseThrow().get("title"));
        assertEquals("My helper", new AgentRuns(chat, new StudioTools(chat, training), data.resolve("agents"))
                .get(id).orElseThrow().get("title"), "the name is kept with the run");
        agents.move(id, "elsewhere");
        assertEquals("elsewhere", agents.get(id).orElseThrow().get("project"));

        java.nio.file.Path away = data.resolve("away-" + id);
        assertEquals(List.of(id), agents.detach(id, away));
        assertTrue(agents.get(id).isEmpty());
        assertTrue(java.nio.file.Files.exists(away.resolve(id + ".json")));
        assertTrue(new AgentRuns(chat, new StudioTools(chat, training), data.resolve("agents")).get(id).isEmpty(),
                "a run taken out stays out after a restart");
        agents.attach(List.of(id), away);
        assertEquals("My helper", agents.get(id).orElseThrow().get("title"));
        assertThrows(IllegalArgumentException.class, () -> agents.attach(List.of(id), away));
    }

    @Test
    void theToolsAreGeneratedFromTheStudiosMethods() {
        Map<String, StudioTools.Entry> tools = agents.tools();
        assertTrue(tools.keySet().containsAll(List.of("list_models", "search_hub", "hub_files", "download_model",
                "list_datasets", "start_training", "stop_training", "list_training_runs")), tools.keySet().toString());
        assertEquals(List.of(), tools.values().stream().filter(e -> e.returns() == null).map(e -> e.tool().name()).toList(),
                "every tool says what it returns");
        assertNull(tools.get("list_datasets").acts());
        assertNotNull(tools.get("download_model").acts());
        assertNotNull(tools.get("start_training").acts());
        Map<String, Object> schema = tools.get("start_training").tool().parameters();
        assertEquals(List.of("model", "data"), schema.get("required"));
        assertTrue(Json.object(schema.get("properties")).containsKey("epochs"));

        String models = agents.call("list_models", Map.of());
        assertTrue(models.contains("\"model\":\"script/bot\""), models);
        assertTrue(agents.call("list_datasets", Map.of()).contains("spirals"));
        assertTrue(agents.call("stop_training", Map.of()).startsWith("error:"), "a missing argument is an error");
        assertThrows(IllegalArgumentException.class, () -> agents.call("format_disk", Map.of()));
    }

    @Test
    void anAgentsReasoningIsAStepOfItsOwn() throws Exception {
        script.turns.add("<think>\nThe user wants the data sets. A tool lists them.\n</think>\n\n"
                + "<tool_call>\n{\"name\": \"list_datasets\", \"arguments\": {}}\n</tool_call>");
        script.turns.add("<think>Spirals is one.</think>\n\nThere are built-in sets such as spirals.");
        Map<String, Object> started = agents.start("Which data sets are there?", "script/bot", "auto", 5, null);
        Map<String, Object> run = until((String) started.get("id"), r -> !"running".equals(r.get("state")));

        assertEquals("done", run.get("state"));
        assertEquals(List.of("goal", "think", "call", "result", "think", "answer"), kinds(run));
        List<Map<String, Object>> steps = Json.array(run.get("steps")).stream().map(Json::object).toList();
        assertEquals("The user wants the data sets. A tool lists them.", steps.get(1).get("text"));
        assertEquals("There are built-in sets such as spirals.", run.get("answer"));
        assertFalse(script.heard.get(1).messages().get(1).text().contains("think"),
                "the reasoning is not sent back to the model");
    }

    @Test
    void anAgentsAnswerCanBeJsonOfASchema() throws Exception {
        script.turns.add("<tool_call>\n{\"name\": \"list_datasets\", \"arguments\": {}}\n</tool_call>");
        script.turns.add("Spirals is one of them.");
        script.turns.add("{\"sets\": [\"spirals\"]}");
        Format format = Format.of(Map.of("format", Map.of("type", "schema", "schema",
                Map.of("type", "object", "properties", Map.of("sets", Map.of("type", "array"))))));
        Map<String, Object> started = agents.start("Which data sets are there?", "script/bot", "auto", 5, null, format);
        Map<String, Object> run = until((String) started.get("id"), r -> !"running".equals(r.get("state")));

        assertEquals("done", run.get("state"));
        assertEquals(List.of("goal", "call", "result", "say", "answer"), kinds(run));
        assertEquals("{\"sets\": [\"spirals\"]}", run.get("answer"));
        assertEquals("schema", Json.object(run.get("format")).get("type"));
        // the turns that may call tools are free text; only the last one is held to the schema
        assertTrue(script.heard.get(0).format().none() && script.heard.get(1).format().none());
        assertEquals("schema", script.heard.get(2).format().kind());
        assertEquals(AgentRuns.SHAPE_ASK, script.heard.get(2).messages().getLast().text());
        // kept with the run, so a reply answers in the same form
        AgentRuns again = new AgentRuns(chat, new StudioTools(chat, training), data.resolve("agents"));
        assertEquals("schema", Json.object(again.get((String) started.get("id")).orElseThrow().get("format")).get("type"));

        assertThrows(IllegalArgumentException.class, () -> agents.start("x", "script/bot", "auto", 5, null,
                Format.of(Map.of("format", Map.of("type", "grammar", "grammar", "root ::= \"a\"")))));
    }

    @Test
    void anAgentCallsToolsAndAnswers() throws Exception {
        script.turns.add("Let me look.\n<tool_call>\n{\"name\": \"list_datasets\", \"arguments\": {}}\n</tool_call>\n"
                + "<tool_response>\nmade up\n</tool_response>");
        script.turns.add("There are built-in sets such as spirals.");
        Map<String, Object> started = agents.start("Which data sets are there?", "script/bot", "ask", 5, null);
        Map<String, Object> run = until((String) started.get("id"), r -> !"running".equals(r.get("state")));

        assertEquals("done", run.get("state"));
        assertEquals("There are built-in sets such as spirals.", run.get("answer"));
        assertEquals("script/bot", run.get("by"));
        assertEquals(List.of("goal", "say", "call", "result", "answer"), kinds(run));
        assertEquals(20L, ((Number) run.get("inputTokens")).longValue());

        ChatRequest first = script.heard.getFirst();
        assertTrue(first.system().contains("<tools>") && first.system().contains("\n- list_models("));
        assertTrue(first.system().contains("# More tools") && first.system().contains("- start_training: ")
                && !first.system().contains("- start_training("), "tools beyond the core ones are listed by name");
        ChatRequest second = script.heard.get(1);
        String asked = second.messages().get(1).text();
        assertFalse(asked.contains("made up"), "what the model imagined after its call is dropped");
        String answered = second.messages().getLast().text();
        assertTrue(answered.startsWith("<tool_response>") && answered.contains("spirals"), answered);
    }

    @Test
    void inAskModeAToolThatActsWaitsForTheUser() throws Exception {
        String call = "<tool_call>{\"name\": \"stop_training\", \"arguments\": {\"id\": \"7\"}}</tool_call>";
        script.turns.add(call);
        script.turns.add("You did not want that, so I left it.");
        String id = (String) agents.start("Stop run 7", "script/bot", "ask", 5, null).get("id");
        Map<String, Object> waiting = until(id, r -> "waiting".equals(r.get("state")));
        Map<String, Object> pending = Json.object(waiting.get("pending"));
        assertEquals("stop_training", pending.get("tool"));
        assertEquals(Map.of("id", "7"), pending.get("arguments"));
        assertFalse(agents.answer("nope", true, null));

        assertTrue(agents.answer(id, false, "keep it going"));
        Map<String, Object> run = until(id, r -> "done".equals(r.get("state")));
        assertTrue(script.heard.get(1).messages().getLast().text().contains("did not allow this call; they said: keep "
                + "it going"));
        assertEquals(List.of("goal", "call", "decision", "result", "answer"), kinds(run));

        // in auto mode it calls the tool itself
        script.turns.add(call);
        script.turns.add("Done.");
        String auto = (String) agents.start("Stop run 7", "script/bot", "auto", 5, null).get("id");
        Map<String, Object> r2 = until(auto, r -> "done".equals(r.get("state")));
        assertEquals(List.of("goal", "call", "result", "answer"), kinds(r2));
        assertTrue(script.heard.getLast().messages().getLast().text().contains("there is no run 7"));
    }

    @Test
    void aScriptCallsToolsThatAreJudgedOneByOne() throws Exception {
        String code = "const sets = tools.list_datasets({});\n"
                + "let stopped;\n"
                + "try { stopped = tools.stop_training({id: '7'}); } catch (e) { stopped = e.name + ': ' + e.message; }\n"
                + "console.log('checked', Object.keys(sets).length > 0);\n"
                + "return {stopped, hasSets: JSON.stringify(sets).includes('spirals')};";
        script.turns.add("<tool_call>" + Json.write(Map.of("name", "run_script", "arguments", Map.of("code", code)))
                + "</tool_call>");
        script.turns.add("Done.");
        String id = (String) agents.start("Check the sets", "script/bot", "ask", 5, null).get("id");
        Map<String, Object> waiting = until(id, r -> "waiting".equals(r.get("state")));
        assertEquals("stop_training", Json.object(waiting.get("pending")).get("tool"), "a call in a script asks too");
        assertTrue(agents.answer(id, false, "not now"));
        Map<String, Object> run = until(id, r -> "done".equals(r.get("state")));
        assertEquals(List.of("goal", "call", "call", "result", "call", "decision", "result", "result", "answer"),
                kinds(run));
        String told = script.heard.getLast().messages().getLast().text();
        assertTrue(told.contains("\"stopped\":\"ToolError: the user did not allow this call; they said: not now\"")
                && told.contains("\"hasSets\":true") && told.contains("\"calls\":2")
                && told.contains("checked true"), told);
        ChatRequest first = script.heard.getFirst();
        assertTrue((first.system() + first.tools()).contains("Returns {result, calls"), "tools say what they return");

        // a script that does not parse says where
        script.turns.add("<tool_call>{\"name\": \"run_script\", \"arguments\": {\"code\": \"let = 1\"}}</tool_call>");
        script.turns.add("Oops.");
        String bad = (String) agents.start("Try", "script/bot", "auto", 5, null).get("id");
        until(bad, r -> "done".equals(r.get("state")));
        String error = script.heard.getLast().messages().getLast().text();
        assertTrue(error.contains("error: line 1"), error);
    }

    @Test
    void findToolsLoadsAToolForTheNextTurns() throws Exception {
        script.turns.add("<tool_call>{\"name\": \"find_tools\", \"arguments\": {\"names\": [\"start_training\"]}}</tool_call>");
        script.turns.add("Loaded.");
        String id = (String) agents.start("Get ready to train", "script/bot", "auto", 5, null).get("id");
        until(id, r -> "done".equals(r.get("state")));
        assertFalse(script.heard.get(0).system().contains("- start_training("));
        assertTrue(script.heard.get(1).system().contains("- start_training(model: string, data: string"), "loaded from the next turn on");
    }

    @Test
    void agentsSeeTheRolesAndWhichTheyMayStart() throws Exception {
        script.turns.add("<tool_call>{\"name\": \"list_roles\", \"arguments\": {}}</tool_call>");
        script.turns.add("Done.");
        String id = (String) agents.start("Which roles are there?", "script/bot", "auto", 5, null).get("id");
        until(id, r -> "done".equals(r.get("state")));
        String told = script.heard.getLast().messages().getLast().text();
        assertTrue(told.contains("\"id\":\"researcher\"") && told.contains("\"purpose\"")
                && told.contains("\"youMayStart\":true"), told);
        assertTrue(script.heard.getFirst().system().contains("list_roles says what each role is for"));
    }

    @Test
    void aChatsAgentCarriesOnWithTheChatsModeAndModel() throws Exception {
        script.turns.add("Hello.");
        String id = (String) agents.start("Say hello", "script/bot", "ask", 5, null).get("id");
        agents.forChat(id, "c1");
        until(id, r -> "done".equals(r.get("state")));
        script.turns.add("Hello again.");
        agents.reply(id, "Once more", "auto", "script/other").orElseThrow();
        Map<String, Object> r = until(id, x -> "done".equals(x.get("state")));
        assertEquals("auto", r.get("mode"));
        assertEquals("script/other", r.get("model"));
        assertEquals("c1", r.get("chat"));
    }

    @Test
    void runsAreKeptStoppedAndContinued() throws Exception {
        script.turns.add("<tool_call>{\"name\": \"download_model\", \"arguments\": {\"repo\": \"a/b\", "
                + "\"option\": \"x.gguf\"}}</tool_call>");
        String id = (String) agents.start("Get a model", "script/bot", "ask", 5, null).get("id");
        until(id, r -> "waiting".equals(r.get("state")));
        assertThrows(IllegalArgumentException.class, () -> agents.delete(id), "a working agent is not removed");
        assertTrue(agents.stop(id));
        Map<String, Object> stopped = until(id, r -> "stopped".equals(r.get("state")));
        assertNull(stopped.get("pending"));

        script.turns.add("Ok, I will not.");
        agents.reply(id, "Never mind").orElseThrow();
        assertEquals("Ok, I will not.", until(id, r -> "done".equals(r.get("state"))).get("answer"));
        String told = script.heard.getLast().messages().getLast().text();
        assertTrue(told.contains("stopped the agent before this call ran") && told.endsWith("Never mind"), told);

        script.turns.add("Hello.");
        String done = (String) agents.start("Say hello", "script/bot", "auto", 5, null).get("id");
        until(done, r -> "done".equals(r.get("state")));
        script.turns.add("Hello again.");
        agents.reply(done, "Once more").orElseThrow();
        Map<String, Object> again = until(done, r -> "done".equals(r.get("state")) && "Hello again.".equals(r.get("answer")));
        assertEquals(4, Json.array(again.get("messages")).size());

        AgentRuns reopened = new AgentRuns(chat, new StudioTools(chat, training), data.resolve("agents"));
        assertEquals(2, reopened.list().size());
        assertEquals(done, reopened.list().getFirst().get("id"));
        assertEquals("Hello again.", reopened.get(done).orElseThrow().get("answer"));
        String third = (String) reopened.start("x", "script/bot", "auto", 1, null).get("id");
        assertEquals("3", third, "ids go on after the kept ones");
        for (int i = 0; i < 200 && "running".equals(reopened.get(third).orElseThrow().get("state")); i++) {
            Thread.sleep(20);   // its last save must land before the folder is cleaned up
        }
        assertTrue(reopened.delete(id));
        assertFalse(reopened.get(id).isPresent());
    }

    @Test
    void anAgentStopsAtItsStepLimitAndOnlySeesItsTools() throws Exception {
        for (int i = 0; i < 3; i++) {
            script.turns.add("<tool_call>{\"name\": \"list_training_runs\", \"arguments\": {}}</tool_call>");
        }
        String id = (String) agents.start("Loop", "script/bot", "auto", 2, List.of("list_training_runs")).get("id");
        Map<String, Object> run = until(id, r -> !"running".equals(r.get("state")));
        assertEquals("out of steps", run.get("state"));
        assertTrue(((String) run.get("error")).contains("all 2 steps"));
        String system = script.heard.getFirst().system();
        assertTrue(system.contains("list_training_runs") && !system.contains("start_training"));

        script.turns.clear();
        script.turns.add("<tool_call>{\"name\": \"start_training\", \"arguments\": {\"model\": \"m\", \"data\": "
                + "\"spirals\"}}</tool_call>");
        script.turns.add("I cannot.");
        String id2 = (String) agents.start("Train", "script/bot", "auto", 3, List.of("list_datasets")).get("id");
        until(id2, r -> "done".equals(r.get("state")));
        assertTrue(script.heard.getLast().messages().getLast().text().contains("error: there is no tool named "
                + "start_training"));
        assertThrows(IllegalArgumentException.class, () -> agents.start("x", "auto", "yolo", 3, null));
        assertThrows(IllegalArgumentException.class, () -> agents.start(" ", "auto", "ask", 3, null));
    }

    @Test
    void anAgentInASubprojectAsksItsParentAndTheAnswerWakesIt() throws Exception {
        io.github.zzuegg.brainiac.studio.project.Projects projects = new io.github.zzuegg.brainiac.studio.project.Projects(
                data.resolve("projects.json"));
        projects.save(new io.github.zzuegg.brainiac.studio.project.Project("film", "Film", "", null, null, null,
                Map.of()), true);
        projects.save(new io.github.zzuegg.brainiac.studio.project.Project("scenes", "Scenes", "", null,
                List.of("list_datasets"), null, Map.of(), Map.of(), List.of("film"),
                io.github.zzuegg.brainiac.studio.project.Project.BY_USER), true);
        AgentRuns runs = new AgentRuns(chat, List.of(new StudioTools(chat, training)), data.resolve("agents2"), projects,
                new Roles(data.resolve("roles")));
        try {
            script.turns.add("<tool_call>{\"name\": \"request_parent\", \"arguments\": {\"kind\": \"question\", "
                    + "\"text\": \"Which colour?\"}}</tool_call>");
            script.turns.add("I asked; I will wait for the answer.");
            script.turns.add("Blue it is.");
            String id = (String) ScopedValue.where(ChatService.SCOPE, projects.get("scenes").orElseThrow())
                    .call(() -> runs.start("Paint the scene", "script/bot", "auto", 6, null)).get("id");
            for (int i = 0; i < 200 && runs.requests().isEmpty(); i++) {
                Thread.sleep(20);
            }
            Map<String, Object> q = runs.requests().getFirst();
            assertEquals("open", q.get("state"));
            assertEquals(List.of("film"), q.get("to"));
            assertEquals("user", q.get("answeredBy"));
            for (int i = 0; i < 200 && "running".equals(runs.get(id).orElseThrow().get("state")); i++) {
                Thread.sleep(20);
            }
            runs.respond((String) q.get("id"), "Blue", null);
            assertThrows(IllegalArgumentException.class, () -> runs.respond((String) q.get("id"), "Red", null),
                    "the first answer counts");
            Map<String, Object> done = null;
            for (int i = 0; i < 200; i++) {
                done = runs.get(id).orElseThrow();
                if ("Blue it is.".equals(done.get("answer"))) {
                    break;
                }
                Thread.sleep(20);
            }
            assertEquals("Blue it is.", done.get("answer"));
            assertTrue(kinds(done).contains("inbox"));
            assertFalse(Boolean.TRUE.equals(done.get("tainted")), "the user's answer is not another agent's word");
            assertTrue(script.heard.getLast().messages().getLast().text().contains("Blue"));

            assertEquals("answered", runs.requests().getFirst().get("state"));
        } finally {
            for (int i = 0; i < 200 && runs.list().stream().anyMatch(r -> "running".equals(r.get("state"))); i++) {
                Thread.sleep(20);
            }
            runs.close();
        }
    }

    @Test
    void agentsReachDownIntoSubprojectsButNotUp() throws Exception {
        io.github.zzuegg.brainiac.studio.project.Projects projects = new io.github.zzuegg.brainiac.studio.project.Projects(
                data.resolve("projects.json"));
        projects.save(new io.github.zzuegg.brainiac.studio.project.Project("film", "Film", "", null, null, null,
                Map.of()), true);
        projects.save(new io.github.zzuegg.brainiac.studio.project.Project("other", "Other", "", null, null, null,
                Map.of()), true);
        AgentRuns runs = new AgentRuns(chat, List.of(new StudioTools(chat, training)), data.resolve("agents3"), projects,
                new Roles(data.resolve("roles")));
        try {
            script.turns.add("<tool_call>{\"name\": \"create_subproject\", \"arguments\": {\"name\": \"Scenes\", "
                    + "\"tools\": [\"list_datasets\"]}}</tool_call>");
            script.turns.add("<tool_call>{\"name\": \"spawn_agent\", \"arguments\": {\"role\": \"assistant\", "
                    + "\"goal\": \"Look\", \"project\": \"other\"}}</tool_call>");
            script.turns.add("<tool_call>{\"name\": \"spawn_agent\", \"arguments\": {\"role\": \"assistant\", "
                    + "\"goal\": \"Look at the data\", \"project\": \"scenes\"}}</tool_call>");
            script.turns.add("Started.");
            script.turns.add("Child done.");
            String id = (String) ScopedValue.where(ChatService.SCOPE, projects.get("film").orElseThrow())
                    .call(() -> runs.start("Make the film", "script/bot", "auto", 8, null)).get("id");
            Map<String, Object> run = null;
            for (int i = 0; i < 300; i++) {
                run = runs.get(id).orElseThrow();
                if (!"running".equals(run.get("state"))) {
                    break;
                }
                Thread.sleep(20);
            }
            assertEquals(List.of("film"), projects.own("scenes").orElseThrow().parents());
            List<Map<String, Object>> results = Json.array(run.get("steps")).stream().map(Json::object)
                    .filter(s -> "result".equals(s.get("kind"))).toList();
            assertEquals(Boolean.FALSE, results.get(1).get("ok"), "not into a project that is not below: " + results.get(1));
            assertEquals(Boolean.TRUE, results.get(2).get("ok"), String.valueOf(results.get(2)));
            String child = (String) Json.array(run.get("children")).getFirst();
            assertEquals("scenes", runs.get(child).orElseThrow().get("project"));
        } finally {
            for (int i = 0; i < 300 && runs.list().stream().anyMatch(r -> "running".equals(r.get("state"))); i++) {
                Thread.sleep(20);
            }
            runs.close();
        }
    }
}
