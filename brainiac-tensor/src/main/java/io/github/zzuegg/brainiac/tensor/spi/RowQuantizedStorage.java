package io.github.zzuegg.brainiac.tensor.spi;

/**
 * Storage of rows kept as 8-bit levels with a scale per row, in a backend's own form (see
 * {@link QuantizedRows} for the rule), written with {@link Backend#writeQuantizedRows}. Only
 * {@link Backend#attentionQuantized} reads it directly; every other operation sees the values
 * through {@link Backend#unpack}.
 */
public interface RowQuantizedStorage extends PackedStorage {
}
