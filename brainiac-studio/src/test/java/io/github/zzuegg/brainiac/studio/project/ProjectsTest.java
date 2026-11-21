package io.github.zzuegg.brainiac.studio.project;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.studio.chat.GpuResidency;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Projects are kept, limit models, tools and files, and route their tasks. */
class ProjectsTest {

    @TempDir
    Path dir;

    @Test
    void aProjectIsKeptWithItsRoutes() throws IOException {
        Path work = Files.createDirectories(dir.resolve("work"));
        Projects ps = new Projects(dir.resolve("projects.json"));
        Project p = ps.save(Project.fromJson(Map.of("name", "Film A", "folder", work.toString(),
                "tools", List.of("make_speech", "read_project_file"), "models", List.of("local/"),
                "routes", Map.of("video", Map.of("model", "local/wan", "policy", "remote", "fallback", "host-gpu/wan")))),
                true);
        assertEquals("film-a", p.id());
        ps.activate(p.id());

        Projects again = new Projects(dir.resolve("projects.json"));
        Project q = again.active();
        assertEquals("Film A", q.name());
        assertEquals(GpuResidency.Policy.REMOTE, q.route("video").policy());
        assertEquals("host-gpu/wan", q.route("video").fallback());
        assertTrue(q.allows("local/qwen"));
        assertTrue(q.allows("host-gpu/wan"));   // where a route sends a task is allowed
        assertFalse(q.allows("openai/gpt-5"));
        assertTrue(q.allowsTool("make_speech"));
        assertFalse(q.allowsTool("start_training"));
        assertEquals(2, again.list().size());
        assertTrue(again.get(Project.DEFAULT).orElseThrow().allows("openai/gpt-5"));
    }

    @Test
    void agentsReadOnlyInsideTheFolder() throws IOException {
        Path work = Files.createDirectories(dir.resolve("work"));
        Files.writeString(work.resolve("notes.txt"), "hi");
        Project p = new Project("p", "P", "", work, null, null, Map.of());
        assertEquals(work.resolve("notes.txt").toAbsolutePath().normalize(), p.inFolder("notes.txt"));
        assertThrows(IllegalArgumentException.class, () -> p.inFolder("../secret.txt"));
        assertThrows(IllegalArgumentException.class, () -> Project.unlimited().inFolder("x"));
    }

    @Test
    void theDefaultProjectStays() throws IOException {
        Projects ps = new Projects(null);
        assertThrows(IllegalArgumentException.class, () -> ps.delete(Project.DEFAULT));
        Project a = ps.save(new Project("a", "A", "", null, null, null, Map.of()), true);
        Project b = ps.save(new Project("a", "A", "", null, null, null, Map.of()), true);
        assertEquals("a-2", b.id());
        ps.activate(a.id());
        ps.delete(a.id());
        assertEquals(Project.DEFAULT, ps.active().id());
    }

    @Test
    void aRunnerIsKeptAndChecked() throws IOException {
        Projects ps = new Projects(dir.resolve("projects.json"));
        ps.save(Project.fromJson(Map.of("name", "B",
                "routes", Map.of("video", Map.of("policy", "remote", "fallback", "vast:video")),
                "runners", Map.of("video", Map.of("gpu", "4090", "maxPrice", 0.6, "hours", 2,
                        "repo", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")))), true);
        Project b = new Projects(dir.resolve("projects.json")).get("b").orElseThrow();
        var r = io.github.zzuegg.brainiac.studio.train.VastRunners.Runner.fromJson(b.runner("video"));
        assertEquals("Wan2.1-T2V-1.3B-Diffusers", r.modelName());
        assertEquals("4090", r.gpu());
        assertTrue(b.allows("vast:video"));
        assertThrows(IllegalArgumentException.class,
                () -> io.github.zzuegg.brainiac.studio.train.VastRunners.Runner.fromJson(Map.of("gpu", "4090")));
    }

    @Test
    void aRunnerFieldLeftEmptyIsNotSet() {
        Map<String, Object> runner = new java.util.HashMap<>(Map.of("maxPrice", 0.5, "repo", ""));
        runner.put("minGpuRam", null);
        Project p = Project.fromJson(Map.of("name", "C", "runners", Map.of("video", runner)));
        assertEquals(Map.of("maxPrice", 0.5), p.runner("video"));
    }

    private static Project sub(String id, List<String> tools, List<String> models, List<String> parents) {
        return new Project(id, id, "", null, tools, models, Map.of(), Map.of(), parents, Project.BY_USER);
    }

    @Test
    void aSubprojectCanBePartOfSeveralAndUsesOnlyWhatEachAllows() throws IOException {
        Projects ps = new Projects(dir.resolve("projects.json"));
        ps.save(sub("film", List.of("make_speech", "web_search", "read_project_file"), List.of("local/", "openai/"), List.of()), true);
        ps.save(sub("research", List.of("web_search", "read_project_file"), List.of("local/qwen"), List.of()), true);
        ps.save(sub("script", null, null, List.of("film", "research")), true);

        Project eff = ps.get("script").orElseThrow();
        assertEquals(List.of("web_search", "read_project_file"), eff.tools());
        assertTrue(eff.allows("local/qwen"));
        assertFalse(eff.allows("local/llama"));
        assertFalse(eff.allows("openai/gpt-5"));
        assertEquals(null, ps.own("script").orElseThrow().tools(), "its own settings are kept as they are");
        assertTrue(ps.within("script", "film"));
        assertFalse(ps.within("film", "script"));
        assertEquals(List.of("script"), ps.descendants("research"));

        assertThrows(IllegalArgumentException.class, () -> ps.save(sub("film", null, null, List.of("script")), false),
                "a project cannot be part of its own subproject");
        assertThrows(IllegalArgumentException.class, () -> ps.save(sub("x", null, null, List.of("nope")), true));
        assertThrows(IllegalArgumentException.class, () -> sub("y", null, null, List.of(Project.DEFAULT)));
        assertEquals(List.of("film", "research"), new Projects(dir.resolve("projects.json")).own("script")
                .orElseThrow().parents(), "the links are kept");
    }

    @Test
    void deletingAParentOnlyCutsSharedSubprojectsLooseAndUndoingPutsThemBack() throws IOException {
        Projects ps = new Projects(null);
        ps.save(sub("a", null, null, List.of()), true);
        ps.save(sub("b", null, null, List.of()), true);
        ps.save(sub("shared", null, null, List.of("a", "b")), true);
        ps.save(sub("only-a", null, null, List.of("a")), true);
        ps.save(sub("deep", null, null, List.of("only-a")), true);

        Projects.Removal r = ps.remove("a");
        assertTrue(ps.own("a").isEmpty());
        assertTrue(ps.own("only-a").isEmpty(), "a subproject of nothing else goes with it");
        assertTrue(ps.own("deep").isEmpty(), "and so does its own");
        assertEquals(List.of("b"), ps.own("shared").orElseThrow().parents(), "a shared one stays, linked to the rest");

        ps.restore(Projects.Removal.fromJson(io.github.zzuegg.brainiac.io.Json.object(
                io.github.zzuegg.brainiac.io.Json.parse(io.github.zzuegg.brainiac.io.Json.write(r.toJson())))));
        assertEquals(List.of("only-a"), ps.own("deep").orElseThrow().parents());
        assertEquals(List.of("b", "a"), ps.own("shared").orElseThrow().parents());
        assertEquals(List.of("a"), ps.own("only-a").orElseThrow().parents());
    }
}
