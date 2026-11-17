// Reading a packed matrix (see VulkanBackend.Packed), included after a push constant block p
// with: InWords q, hi, scaleCodes, offsetCodes, scales, offsets; uint n, group, flags,
// perSuper; float shift. flags: 1 nibbles, 2 offsets, 4 scale codes, 8 offset codes,
// 16 float16 scales and offsets, 32 4-bit indices (as nibbles) into a table of 16 signed bytes in hi; bits 8-11 the high bits per level kept in the hi plane,
// bits 12-15 the bits per level.

uint ubyteAt(InWords w, uint b) {
    return bitfieldExtract(w.v[b >> 2], int((b & 3u) * 8u), 8);
}

int tableAt(uint i) {
    return bitfieldExtract(int(p.hi.v[i >> 2]), int((i & 3u) * 8u), 8);
}

// the level at row r, column c
int levelAt(uint r, uint c) {
    if ((p.flags & 32u) != 0) {
        uint b = (r >> 1) * p.n + c;
        return tableAt(bitfieldExtract(p.q.v[b >> 2], int((b & 3u) * 8u + (r & 1u) * 4u), 4));
    }
    uint planeBits = (p.flags >> 8) & 15u;
    if (planeBits != 0) {
        uint lb = (r >> 1) * p.n + c;
        uint low = bitfieldExtract(p.q.v[lb >> 2], int((lb & 3u) * 8u + (r & 1u) * 4u), 4);
        uint perByte = 8u / planeBits;
        uint hb = (r / perByte) * p.n + c;
        uint high = bitfieldExtract(p.hi.v[hb >> 2], int((hb & 3u) * 8u + (r % perByte) * planeBits), int(planeBits));
        return int(low | (high << 4)) - (1 << (int((p.flags >> 12) & 15u) - 1));
    }
    if ((p.flags & 1u) != 0) {
        uint b = (r >> 1) * p.n + c;
        return bitfieldExtract(int(p.q.v[b >> 2]), int((b & 3u) * 8u + (r & 1u) * 4u), 4);
    }
    uint b = r * p.n + c;
    return bitfieldExtract(int(p.q.v[b >> 2]), int((b & 3u) * 8u), 8);
}

// the levels of columns c .. c + 3 at row r, c a multiple of 4 and n of 4 (one word each)
ivec4 levels4At(uint r, uint c) {
    if ((p.flags & 32u) != 0) {
        uint w = p.q.v[((r >> 1) * p.n + c) >> 2];
        int s = int((r & 1u) * 4u);
        return ivec4(tableAt(bitfieldExtract(w, s, 4)), tableAt(bitfieldExtract(w, 8 + s, 4)),
                     tableAt(bitfieldExtract(w, 16 + s, 4)), tableAt(bitfieldExtract(w, 24 + s, 4)));
    }
    uint planeBits = (p.flags >> 8) & 15u;
    if (planeBits != 0) {
        uint lw = p.q.v[((r >> 1) * p.n + c) >> 2];
        int ls = int((r & 1u) * 4u);
        uint perByte = 8u / planeBits;
        uint hw = p.hi.v[((r / perByte) * p.n + c) >> 2];
        int hs = int((r % perByte) * planeBits);
        int pb = int(planeBits);
        uvec4 u = uvec4(bitfieldExtract(lw, ls, 4) | (bitfieldExtract(hw, hs, pb) << 4),
                        bitfieldExtract(lw, 8 + ls, 4) | (bitfieldExtract(hw, 8 + hs, pb) << 4),
                        bitfieldExtract(lw, 16 + ls, 4) | (bitfieldExtract(hw, 16 + hs, pb) << 4),
                        bitfieldExtract(lw, 24 + ls, 4) | (bitfieldExtract(hw, 24 + hs, pb) << 4));
        return ivec4(u) - (1 << (int((p.flags >> 12) & 15u) - 1));
    }
    uint shift = 0;
    uint b = r * p.n + c;
    int bits = 8;
    if ((p.flags & 1u) != 0) {
        b = (r >> 1) * p.n + c;
        shift = (r & 1u) * 4u;
        bits = 4;
    }
    int w = int(p.q.v[b >> 2]);
    return ivec4(bitfieldExtract(w, int(shift), bits), bitfieldExtract(w, int(8 + shift), bits),
                 bitfieldExtract(w, int(16 + shift), bits), bitfieldExtract(w, int(24 + shift), bits));
}

float superAt(InWords w, uint i) {
    if ((p.flags & 16u) != 0) {
        return unpackHalf2x16(w.v[i >> 1])[i & 1u];
    }
    return uintBitsToFloat(w.v[i]);
}

float codeAt(InWords w, uint i) {
    return float(bitfieldExtract(int(w.v[i >> 2]), int((i & 3u) * 8u), 8));
}

// the scale of group g in column c
float scaleAt(uint g, uint c) {
    float s = superAt(p.scales, (g / p.perSuper) * p.n + c);
    return (p.flags & 4u) != 0 ? codeAt(p.scaleCodes, g * p.n + c) * s : s;
}

// the offset of group g in column c, whose scale is s
float offsetAt(uint g, uint c, float s) {
    float o = superAt(p.offsets, (g / p.perSuper) * p.n + c);
    o = (p.flags & 8u) != 0 ? codeAt(p.offsetCodes, g * p.n + c) * o : o;
    return o + s * p.shift;
}
