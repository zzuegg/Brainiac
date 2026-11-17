package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Sequence packing with document masking: many short documents share one fixed-length
 * training row instead of each being padded to the row length, and every token carries its
 * document id and its position inside its document. A model that masks attention across
 * documents and restarts positions (as {@code HookedDecoder.forward(tokens, positions,
 * segments, training)} does, the "intra-document masking" of Llama 3 and Krell et al.
 * 2021, <a href="https://arxiv.org/abs/2107.02027">arXiv:2107.02027</a>) computes exactly the
 * same loss as on the unpacked documents, with far less padding.
 *
 * <p>Documents are placed first-fit decreasing; one longer than a row is cut into
 * row-sized pieces.
 */
public final class SequencePacker {

    private SequencePacker() {
    }

    /**
     * Packed rows.
     *
     * @param tokens {@code [rows][length]} inputs, padding where unused
     * @param targets {@code [rows][length]} the next token of the same document, -1 at a
     *     document's last token and in padding
     * @param positions {@code [rows][length]} the position inside the document
     * @param segments {@code [rows][length]} a document id per token, unique per row
     *     (-1 for padding)
     */
    public record Packed(int[][] tokens, int[][] targets, int[][] positions, int[][] segments) {

        /** @return the share of slots holding real tokens */
        public double efficiency() {
            long used = 0;
            long all = 0;
            for (int r = 0; r < tokens.length; r++) {
                for (int t = 0; t < tokens[r].length; t++) {
                    all++;
                    used += segments[r][t] >= 0 ? 1 : 0;
                }
            }
            return all == 0 ? 0 : (double) used / all;
        }
    }

    /**
     * Packs documents.
     *
     * @param documents token sequences, each at least 1 token
     * @param length the row length
     * @param pad the padding token
     * @return the rows
     */
    public static Packed pack(List<int[]> documents, int length, int pad) {
        List<int[]> pieces = new ArrayList<>();
        for (int[] d : documents) {
            for (int i = 0; i < d.length; i += length) {
                pieces.add(Arrays.copyOfRange(d, i, Math.min(d.length, i + length)));
            }
        }
        pieces.sort(Comparator.comparingInt((int[] p) -> p.length).reversed());
        List<List<int[]>> rows = new ArrayList<>();
        List<Integer> free = new ArrayList<>();
        for (int[] p : pieces) {
            int at = -1;
            for (int r = 0; r < rows.size(); r++) {
                if (free.get(r) >= p.length) {
                    at = r;
                    break;
                }
            }
            if (at < 0) {
                rows.add(new ArrayList<>());
                free.add(length);
                at = rows.size() - 1;
            }
            rows.get(at).add(p);
            free.set(at, free.get(at) - p.length);
        }
        int n = rows.size();
        int[][] tokens = new int[n][length];
        int[][] targets = new int[n][length];
        int[][] positions = new int[n][length];
        int[][] segments = new int[n][length];
        for (int r = 0; r < n; r++) {
            Arrays.fill(tokens[r], pad);
            Arrays.fill(targets[r], -1);
            Arrays.fill(segments[r], -1);
            int at = 0;
            int seg = 0;
            for (int[] p : rows.get(r)) {
                for (int i = 0; i < p.length; i++) {
                    tokens[r][at + i] = p[i];
                    positions[r][at + i] = i;
                    segments[r][at + i] = seg;
                    targets[r][at + i] = i + 1 < p.length ? p[i + 1] : -1;
                }
                at += p.length;
                seg++;
            }
            for (int t = at; t < length; t++) {
                positions[r][t] = t - at;
            }
        }
        return new Packed(tokens, targets, positions, segments);
    }

    /**
     * Pads each document to its own row instead, for comparison.
     *
     * @param documents token sequences
     * @param length the row length (longer documents are cut)
     * @param pad the padding token
     * @return one row per document piece
     */
    public static Packed padded(List<int[]> documents, int length, int pad) {
        List<int[]> rows = new ArrayList<>();
        for (int[] d : documents) {
            for (int i = 0; i < d.length; i += length) {
                rows.add(Arrays.copyOfRange(d, i, Math.min(d.length, i + length)));
            }
        }
        int[][] tokens = new int[rows.size()][];
        int[][] targets = new int[rows.size()][];
        int[][] positions = new int[rows.size()][];
        int[][] segments = new int[rows.size()][];
        for (int r = 0; r < rows.size(); r++) {
            Packed one = pack(List.of(rows.get(r)), length, pad);
            tokens[r] = one.tokens()[0];
            targets[r] = one.targets()[0];
            positions[r] = one.positions()[0];
            segments[r] = one.segments()[0];
        }
        return new Packed(tokens, targets, positions, segments);
    }
}
