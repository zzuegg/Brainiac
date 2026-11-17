package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

class PreferenceTrainingTest {

    /** A prompt, then an answer the model learns. */
    static InstructionData.Example example(int[] prompt, int[] answer) {
        int[] tokens = Arrays.copyOf(prompt, prompt.length + answer.length);
        System.arraycopy(answer, 0, tokens, prompt.length, answer.length);
        boolean[] train = new boolean[tokens.length];
        Arrays.fill(train, prompt.length, tokens.length, true);
        return new InstructionData.Example(tokens, train);
    }

    static List<PreferenceTraining.Pair> pairs() {
        List<PreferenceTraining.Pair> pairs = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            int[] prompt = {1, 2 + i, 3 + i};
            pairs.add(new PreferenceTraining.Pair(example(prompt, new int[] {10, 11, 12 + i}),
                    example(prompt, new int[] {20 + i, 21})));
        }
        return pairs;
    }

    @Test
    void sequenceLogProbsAreTheAnswersLikelihood() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 41, GeneratorTest.DEVICE, DType.FLOAT32);
        List<PreferenceTraining.Pair> pairs = pairs();
        float[][] ref = PreferenceTraining.reference(model, pairs, 3);
        for (int i = 0; i < pairs.size(); i++) {
            int[] prompt = Arrays.copyOf(pairs.get(i).chosen().tokens(), 3);
            int[] chosen = Arrays.copyOfRange(pairs.get(i).chosen().tokens(), 3, 6);
            int[] rejected = Arrays.copyOfRange(pairs.get(i).rejected().tokens(), 3, 5);
            assertEquals(Evaluation.logLikelihood(model, prompt, chosen), ref[i][0], 1e-4);
            assertEquals(Evaluation.logLikelihood(model, prompt, rejected), ref[i][1], 1e-4);
        }
    }

    @Test
    void lossesStartWhereTheFormulasSay() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 42, GeneratorTest.DEVICE, DType.FLOAT32);
        List<PreferenceTraining.Pair> pairs = pairs();
        float[][] ref = PreferenceTraining.reference(model, pairs, 4);
        // the model is its own reference: h = 0
        PreferenceTraining.Step dpo = PreferenceTraining.loss(model, pairs, ref, PreferenceTraining.Loss.DPO, 0.1f);
        assertEquals(Math.log(2), dpo.loss().value().item(), 1e-4);
        assertArrayEquals(new float[4], dpo.chosenRewards(), 1e-4f);
        assertEquals(25f, PreferenceTraining.loss(model, pairs, ref, PreferenceTraining.Loss.IPO, 0.1f)
                .loss().value().item(), 1e-3f);
        assertEquals(1f, PreferenceTraining.loss(model, pairs, ref, PreferenceTraining.Loss.HINGE, 0.1f)
                .loss().value().item(), 1e-4f);
    }

    @Test
    void dpoWithLoraLearnsThePreference() {
        Llama model = Llama.random(LlamaConfig.small(40, 32, 2, 4, 2), 43, GeneratorTest.DEVICE, DType.FLOAT32);
        List<Var> params = model.addLora(4, 8, name -> true, 7);
        List<PreferenceTraining.Pair> pairs = pairs();
        float[][] ref = PreferenceTraining.reference(model, pairs, 4);
        double first = Double.NaN;
        PreferenceTraining.Step s = null;
        for (int step = 0; step < 30; step++) {
            s = PreferenceTraining.loss(model, pairs, ref, PreferenceTraining.Loss.DPO, 0.5f);
            if (step == 0) {
                first = s.loss().value().item();
            }
            s.loss().backward();
            for (Var p : params) {
                if (p.grad() != null) {
                    p.assign(p.value().sub(p.grad().mul(0.5f)));
                    p.zeroGrad();
                }
            }
        }
        assertTrue(s.loss().value().item() < first - 0.3, first + " -> " + s.loss().value().item());
        assertEquals(1.0, s.accuracy());
        assertTrue(s.margin() > 0);
    }
}
