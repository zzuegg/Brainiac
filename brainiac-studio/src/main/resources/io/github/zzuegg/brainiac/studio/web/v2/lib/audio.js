// Sound in the browser: the peaks of a recording for a waveform, and any recording as a short
// mono WAV for the studio (a new voice is made from one).

const peaks = new Map();

/**
 * The loudness of a recording in `bars` slices, 0 to 1, worked out once per key.
 * @param {string} key names the recording, such as its path
 * @param {() => Promise<Blob>} read reads it
 * @param {number} bars how many slices
 * @returns {Promise<{peaks: number[], seconds: number}>}
 */
export function peaksOf(key, read, bars = 96) {
  let p = peaks.get(key);
  if (!p) {
    p = read().then(b => b.arrayBuffer()).then(async buf => {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      let d;
      try { d = await ctx.decodeAudioData(buf); } finally { ctx.close?.(); }
      const ch = d.getChannelData(0), step = Math.max(1, Math.floor(ch.length / bars)), out = [];
      for (let b = 0; b < bars; b++) {
        let m = 0;
        for (let i = b * step, end = Math.min(ch.length, i + step); i < end; i += 4) m = Math.max(m, Math.abs(ch[i]));
        out.push(m);
      }
      const top = Math.max(0.01, ...out);
      return { peaks: out.map(v => v / top), seconds: d.duration };
    });
    p.catch(() => peaks.delete(key));
    peaks.set(key, p);
  }
  return p;
}

/**
 * A recording as 16-bit mono WAV at a rate, base64.
 * @param {Blob} blob any audio or video the browser can read
 * @param {number} rate samples per second
 * @returns {Promise<{data: string, seconds: number}>}
 */
export async function toWav(blob, rate = 24000) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try { decoded = await ctx.decodeAudioData(await blob.arrayBuffer()); } finally { ctx.close?.(); }
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * rate)), rate);
  const src = off.createBufferSource();
  src.buffer = decoded; src.connect(off.destination); src.start();
  const pcm = (await off.startRendering()).getChannelData(0);
  const v = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  str(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); str(8, "WAVE"); str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 0x7fff, true);
  const bytes = new Uint8Array(v.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return { data: btoa(bin), seconds: decoded.duration };
}

/** Seconds as m:ss. */
export const clock = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
