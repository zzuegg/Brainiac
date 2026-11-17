package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import java.util.List;
import org.junit.jupiter.api.Test;

class ToolRuntimeTest {

    @Test
    void calculatorFollowsPrecedence() {
        assertEquals("84", ToolRuntime.calculate("12*7"));
        assertEquals("14", ToolRuntime.calculate("2+3*4"));
        assertEquals("-5", ToolRuntime.calculate("3-8"));
        assertEquals("?", ToolRuntime.calculate("3+"));
        assertEquals("?", ToolRuntime.calculate("[3"));
    }

    @Test
    void trainingChunksSkipToolOutput() {
        CharacterTokenizer tok = new CharacterTokenizer("0123456789*=[]>#");
        ToolRuntime rt = ToolRuntime.of(tok, ToolRuntime.Tool.CALCULATOR, '[', '>', ']');
        List<int[][]> chunks = rt.trainingChunks(tok.encode("3*4="), tok.encode("[3*4>12]12#"));
        assertEquals(2, chunks.size());
        assertArrayEquals(tok.encode("3*4="), chunks.get(0)[0]);
        assertArrayEquals(tok.encode("[3*4>"), chunks.get(0)[1]);
        assertArrayEquals(tok.encode("3*4=[3*4>12]"), chunks.get(1)[0]);
        assertArrayEquals(tok.encode("12#"), chunks.get(1)[1]);
    }
}
