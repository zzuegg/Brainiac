package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.tensor.DType;

import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;

class GroupPolicyOptimizationTest {

    static double share(String text, char c) {
        return text.isEmpty() ? 0 : text.chars().filter(x -> x == c).count() / (double) text.length();
    }

    @Test
    void rewardsGoUpWhereTheVerifierPoints() {
        String chars = "abcd";
        Llama model = Llama.random(LlamaConfig.small(chars.length(), 32, 2, 4, 2), 61, GeneratorTest.DEVICE,
                DType.FLOAT32);
        Generator gen = new Generator(model, new CharacterTokenizer(chars));
        GroupPolicyOptimization grpo = new GroupPolicyOptimization(gen, 6,
                Generator.Options.sampled(8, Sampling.temperature(1f), 3).withStopTokens(Set.of()),
                (prompt, answer) -> share(answer, 'a'));
        List<Var> params = model.parameters();
        double first = 0;
        double last = 0;
        for (int step = 0; step < 25; step++) {
            GroupPolicyOptimization.Step s = grpo.step(model, List.of("b", "cd"));
            assertEquals(2, s.answers().size());
            assertEquals(6, s.answers().getFirst().size());
            if (step < 3) {
                first += s.meanReward() / 3;
            }
            if (step >= 22) {
                last += s.meanReward() / 3;
            }
            if (!s.loss().requiresGrad()) {
                continue;   // every group scored the same
            }
            s.loss().backward();
            for (Var p : params) {
                if (p.grad() != null) {
                    p.assign(p.value().sub(p.grad().mul(0.5f)));
                    p.zeroGrad();
                }
            }
        }
        assertTrue(last > first + 0.2, "mean reward " + first + " -> " + last);
    }

    @Test
    void equalRewardsGiveNoGradient() {
        Llama model = Llama.random(LlamaConfig.small(4, 32, 2, 4, 2), 62, GeneratorTest.DEVICE, DType.FLOAT32);
        Generator gen = new Generator(model, new CharacterTokenizer("abcd"));
        GroupPolicyOptimization grpo = new GroupPolicyOptimization(gen, 3,
                Generator.Options.sampled(4, Sampling.temperature(1f), 3).withStopTokens(Set.of()), (p, a) -> 1);
        GroupPolicyOptimization.Step s = grpo.step(model, List.of("a"));
        assertEquals(0f, s.loss().value().item(), 0f);
        assertFalse(s.loss().requiresGrad());
        assertEquals(1.0, s.meanReward(), 0);
    }
}
