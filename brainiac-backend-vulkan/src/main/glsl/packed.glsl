// Packed weights (see PackedMatrix): levels in a byte buffer read as 32-bit words, one
// signed byte per value row-major, or with nibbles != 0 two per byte, the byte at
// (r / 2) * n + c holding row r (even) in its low four bits and row r + 1 in its high four;
// one float scale per column for every `group` rows.
layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer InWords { uint v[]; };

// the byte at position b, as a signed value
int packed_byte(InWords q, uint b) {
    return bitfieldExtract(int(q.v[b >> 2]), int((b & 3u) * 8u), 8);
}
