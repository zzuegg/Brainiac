package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.io.ChatTemplate;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.tensor.DType;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class InstructionDataTest {

    private static final PretrainedTokenizer TOK = AgentTest.tokenizer();

    private static String trainedText(InstructionData.Example e) {
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < e.tokens().length; i++) {
            if (e.train()[i]) {
                ids.add(e.tokens()[i]);
            }
        }
        return TOK.decode(ids.stream().mapToInt(Integer::intValue).toArray(), false);
    }

    @Test
    void onlyTheAssistantsRepliesAreLearned() {
        List<Map<String, Object>> chat = List.of(ChatTemplate.message("system", "Be brief."),
                ChatTemplate.message("user", "Hi"), ChatTemplate.message("assistant", "Hello!"),
                ChatTemplate.message("user", "Bye"), ChatTemplate.message("assistant", "Ciao."));
        InstructionData.Example e = InstructionData.encode(TOK, chat);
        assertEquals(TOK.applyChatTemplate(chat, false), TOK.decode(e.tokens(), false));
        assertEquals("Hello!\nCiao.\n", trainedText(e));
        // the prompt before a reply is tokenized as when generating it
        int[] prompt = TOK.encode(TOK.applyChatTemplate(chat.subList(0, 2), true), true);
        assertArrayEquals(prompt, java.util.Arrays.copyOf(e.tokens(), prompt.length));

        InstructionData.Example c = InstructionData.completion(TOK, "2+2=", "4");
        assertEquals("4<end>", trainedText(c));
    }

    @Test
    void paddingAndPackingLearnTheSameTokens() {
        Llama model = Llama.random(LlamaConfig.small(97, 32, 2, 4, 2), 5, TestDevice.get(), DType.FLOAT32);
        List<InstructionData.Example> examples = List.of(
                InstructionData.completion(TOK, "Q: one", " A: 1"),
                InstructionData.completion(TOK, "Q: two two", " A: 2"),
                InstructionData.completion(TOK, "Q: x", " A: three"));
        List<InstructionData.Batch> padded = InstructionData.pad(examples, 3, 64);
        List<InstructionData.Batch> packed = InstructionData.pack(examples, 1, 64);
        assertEquals(1, padded.size());
        assertEquals(1, packed.size());
        assertEquals(1, packed.getFirst().tokens().length);
        float a = model.loss(padded.getFirst()).value().item();
        float b = model.loss(packed.getFirst()).value().item();
        assertEquals(a, b, 1e-4f);

        // the same as each example on its own, weighted by its trained tokens
        double sum = 0;
        int count = 0;
        for (InstructionData.Example e : examples) {
            float one = model.loss(InstructionData.pad(List.of(e), 1, 64).getFirst()).value().item();
            int n = e.trained();
            sum += one * n;
            count += n;
        }
        assertEquals(sum / count, a, 1e-4);

        // short rows: examples move to new rows and long ones are cut
        List<InstructionData.Batch> tight = InstructionData.pack(examples, 2, 14);
        assertEquals(2, tight.size());
        assertEquals(14, tight.getFirst().tokens()[0].length);
        assertThrows(IllegalArgumentException.class, () -> InstructionData.pack(examples, 0, 10));
    }

    @Test
    void gradientsReachTheWeights() {
        Llama model = Llama.random(LlamaConfig.small(97, 32, 1, 4, 2), 6, TestDevice.get(), DType.FLOAT32);
        InstructionData.Batch b = InstructionData.pack(List.of(InstructionData.completion(TOK, "a", "b"),
                InstructionData.completion(TOK, "c", "d")), 1, 16).getFirst();
        var loss = model.loss(b);
        loss.backward();
        double norm = 0;
        for (var p : model.parameters()) {
            if (p.grad() != null) {
                for (float g : p.grad().toArray()) {
                    norm += g * g;
                }
            }
        }
        org.junit.jupiter.api.Assertions.assertTrue(norm > 0);
    }
}
