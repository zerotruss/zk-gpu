// Generates golden test vectors with ffjavascript, an implementation independent of arkworks
// and of this library. Deterministic: same seed, same files.
//   npm install && node gen.mjs
import { getCurveFromName } from "ffjavascript";
import { writeFileSync } from "node:fs";

const curve = await getCurveFromName("bn128", true);
const { Fr, G1 } = curve;
const r = Fr.p;
const outDir = new URL("../../tests/golden/", import.meta.url);

// xorshift64*, deterministic
let state = 0x9e3779b97f4a7c15n;
const next64 = () => {
  state ^= state >> 12n; state ^= (state << 25n) & ((1n << 64n) - 1n); state ^= state >> 27n;
  return (state * 0x2545f4914f6cdd1dn) & ((1n << 64n) - 1n);
};
const randFr = () => ((next64() << 192n) | (next64() << 128n) | (next64() << 64n) | next64()) % r;

const hex = (x) => x.toString(16);
const le32 = (x) => Uint8Array.from({ length: 32 }, (_, i) => Number((x >> BigInt(8 * i)) & 255n));
const affine = (k) => G1.toAffine(G1.timesScalar(G1.g, k));
const coords = (buf) => { const [x, y, z] = G1.toObject(G1.toAffine(buf)); return z === 0n ? ["0", "0"] : [hex(x), hex(y)]; };

// ---- NTT
async function nttCase(logN, direction) {
  const n = 1 << logN;
  const input = Array.from({ length: n }, randFr);
  const buf = new Uint8Array(n * 32);
  input.forEach((v, i) => buf.set(Fr.e(v), i * 32));
  const out = direction === "forward" ? await Fr.fft(buf) : await Fr.ifft(buf);
  const output = Array.from({ length: n }, (_, i) => hex(Fr.toObject(out.slice(i * 32, (i + 1) * 32))));
  return { log_n: logN, direction, input: input.map(hex), output };
}
const ntt = [await nttCase(10, "forward"), await nttCase(10, "inverse"), await nttCase(12, "forward")];
writeFileSync(new URL("ntt.json", outDir), JSON.stringify({ generator: "ffjavascript 0.3.1 (bn128)", cases: ntt }));

// ---- MSM
async function msmCase(name, kScalars, scalars, infinityAt = []) {
  const n = scalars.length;
  const bases = new Uint8Array(n * 64);
  const baseCoords = [];
  kScalars.forEach((k, i) => {
    const p = infinityAt.includes(i) ? G1.zeroAffine : affine(k);
    bases.set(p, i * 64);
    baseCoords.push(coords(p));
  });
  const sc = new Uint8Array(n * 32);
  scalars.forEach((s, i) => sc.set(le32(s), i * 32));
  const result = await G1.multiExpAffine(bases, sc);
  return { name, bases: baseCoords, scalars: scalars.map(hex), result: coords(result), bases_buf: bases, scalars_buf: sc };
}

// Independent self-check of ffjavascript's own MSM against a plain sum of scalar multiples.
function naiveMsm(c) {
  let acc = G1.zero;
  c.bases.forEach((_, i) => {
    const p = c.bases_buf.slice(i * 64, (i + 1) * 64);
    acc = G1.add(acc, G1.timesScalar(p, BigInt("0x" + c.scalars[i])));
  });
  return coords(acc);
}

const nRandom = 2000;
const random = await msmCase("random", Array.from({ length: nRandom }, randFr), Array.from({ length: nRandom }, randFr));

// Awkward shapes: boundary scalars, infinities, repeated points, P and -P with equal scalars.
const nEdge = 1500;
const special = [0n, 1n, 2n, r - 1n, r - 2n, (1n << 128n) - 1n, 1n << 128n, (1n << 253n) - 1n, (1n << 252n), 1n << 63n, (1n << 64n) + 1n];
const edgeK = Array.from({ length: nEdge }, (_, i) => (i % 5 === 0 ? 777n : randFr()));      // many repeats of one point
edgeK[10] = r - 777n;                                                                          // = -(777 G)
const edgeS = Array.from({ length: nEdge }, (_, i) => (i % 3 === 0 ? special[i % special.length] : randFr()));
edgeS[10] = edgeS[5];                                                                          // P*s + (-P)*s cancels
const edge = await msmCase("edge", edgeK, edgeS, [3, 100, 101]);

// Small enough to cross-check against the naive sum.
const small = await msmCase("small", Array.from({ length: 60 }, randFr), Array.from({ length: 60 }, randFr), [4]);
const smallEdge = await msmCase("small_edge", [5n, r - 5n, 9n, 9n, 11n], [7n, 7n, r - 1n, 1n, 0n]);
for (const c of [small, smallEdge]) {
  const ok = JSON.stringify(naiveMsm(c)) === JSON.stringify(c.result);
  console.log(`self-check ${c.name}: ffjavascript MSM ${ok ? "==" : "!="} naive sum`);
  if (!ok) process.exit(1);
}
const cases = [random, edge, small, smallEdge].map(({ bases_buf, scalars_buf, ...c }) => c);
writeFileSync(new URL("msm.json", outDir), JSON.stringify({ generator: "ffjavascript 0.3.1 (bn128)", cases }));
console.log("wrote", ntt.length, "NTT cases and", cases.length, "MSM cases:", cases.map((c) => `${c.name}(n=${c.scalars.length})`).join(", "));
await curve.terminate();
