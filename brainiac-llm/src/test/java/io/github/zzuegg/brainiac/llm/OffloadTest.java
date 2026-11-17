package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.nio.file.Path;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Models split between the CPU and the test device: the same results as on one device. */
class OffloadTest {

    private static Path resource(String name) {
        return LlamaTest.resource(name);
    }

    @ParameterizedTest
    @ValueSource(strings = {"qwen2", "gemma3", "llama"})
    void splitModelsGiveTheSameResults(String name) {
        Device target = TestDevice.get();
        int[][] prompt = {{1, 5, 9, 3, 7}, {2, 4, 6, 8, 10}};
        Llama whole = Llama.load(resource(name), Device.cpu(), DType.FLOAT32);
        Llama split = Llama.load(resource(name), Device.cpu(), DType.FLOAT32)
                .offload(target, DType.FLOAT32, whole.config().layers() / 2);
        assertArrayEquals(whole.forward(prompt).value().toArray(), split.forward(prompt).value().toArray(), 1e-4f);

        // decoding through the cache, one step after the prompt
        LanguageModel.State a = whole.newState(2);
        LanguageModel.State b = split.newState(2);
        assertArrayEquals(whole.next(a, prompt, null).toArray(), split.next(b, prompt, null).toArray(), 1e-4f);
        int[][] step = {{11}, {12}};
        assertArrayEquals(whole.next(a, step, null).toArray(), split.next(b, step, null).toArray(), 1e-4f);
        LanguageModel.State c = b.select(new int[] {1}).join(split.newState(1));
        assertEquals(2, split.next(c, new int[][] {{3}, {3}}, new boolean[][] {{false}, {false}})
                .shape().dim(0));

        // training: the loss and its gradients cross the split
        Var lw = whole.loss(prompt);
        Var ls = split.loss(prompt);
        assertEquals(lw.value().item(), ls.value().item(), 1e-4f);
        lw.backward();
        ls.backward();
        String last = "model.layers." + (whole.config().layers() - 1) + ".mlp.down_proj.weight";
        String first = "model.layers.0.input_layernorm.weight";
        for (String p : new String[] {first, last}) {
            if (whole.namedParameters().containsKey(p)) {
                assertArrayEquals(whole.namedParameters().get(p).grad().toArray(),
                        split.namedParameters().get(p).grad().toArray(), 1e-4f, p);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {-1, 3})
    void refusesImpossibleSplits(int layers) {
        Llama model = Llama.load(resource("qwen2"), Device.cpu(), DType.FLOAT32);
        assertThrows(IllegalArgumentException.class, () -> model.offload(TestDevice.get(), DType.FLOAT32, layers));
    }
}
