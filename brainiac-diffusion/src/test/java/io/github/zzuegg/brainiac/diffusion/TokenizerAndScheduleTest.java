package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.Json;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** The tokenizer and the noise schedules against transformers' and diffusers' results. */
class TokenizerAndScheduleTest {

    private static Path dir() throws Exception {
        return Path.of(TokenizerAndScheduleTest.class.getResource("/tiny/reference.safetensors").toURI()).getParent();
    }

    @Test
    void tokenizerMatchesTransformers() throws Exception {
        // vocab.json and merges.txt, and tokenizer.json as diffusers saves it
        for (ClipTokenizer[] t : new ClipTokenizer[][] {
                {ClipTokenizer.load(dir().resolve("vocab"), 16), ClipTokenizer.load(dir().resolve("vocab"), 16, "!")},
                {ClipTokenizer.load(dir().resolve("sdxl/tokenizer"), 16),
                        ClipTokenizer.load(dir().resolve("sdxl/tokenizer_2"), 16)}}) {
            for (Object o : Json.array(Json.read(dir().resolve("tokenizer_cases.json")))) {
                Map<String, Object> c = Json.object(o);
                String prompt = (String) c.get("prompt");
                assertArrayEquals(ints(c.get("ids")), t[0].encode(prompt), prompt);
                assertArrayEquals(ints(c.get("ids_pad_bang")), t[1].encode(prompt), prompt + " (pad !)");
            }
        }
    }

    private static int[] ints(Object list) {
        return Json.array(list).stream().mapToInt(v -> ((Number) v).intValue()).toArray();
    }

    @Test
    void schedulesMatchDiffusers() throws Exception {
        for (Object o : Json.array(Json.read(dir().resolve("schedules.json")))) {
            Map<String, Object> c = Json.object(o);
            String spacing = (String) c.get("spacing");
            NoiseSchedule schedule = NoiseSchedule.fromJson(Map.of("timestep_spacing", spacing, "steps_offset", 1));
            int steps = ((Number) c.get("steps")).intValue();
            NoiseSchedule.Plan plan = schedule.plan(steps, (Boolean) c.get("karras"));
            String what = spacing + " karras=" + c.get("karras") + " steps=" + steps;
            List<Object> ts = Json.array(c.get("timesteps"));
            List<Object> sigmas = Json.array(c.get("sigmas"));
            assertEquals(ts.size(), plan.steps(), what);
            for (int i = 0; i < ts.size(); i++) {
                assertEquals(((Number) ts.get(i)).doubleValue(), plan.timesteps()[i], 2e-3, what + " timestep " + i);
            }
            for (int i = 0; i < sigmas.size(); i++) {
                double e = ((Number) sigmas.get(i)).doubleValue();
                assertEquals(e, plan.sigmas()[i], 1e-4 * Math.max(1, e), what + " sigma " + i);
            }
            assertEquals(((Number) c.get("init")).doubleValue(), plan.initialSigma(), 1e-3, what);
        }
    }
}
