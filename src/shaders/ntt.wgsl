// One pass of a decimation-in-frequency NTT: `s` consecutive radix-2 stages done in
// workgroup memory on a tile of 512 elements. Elements that interact across the pass are
// `stride = 2^lm` apart; adjacent threads take adjacent columns so global accesses coalesce.
//
// Reads `src`, writes `dst` (never the same buffer: the bit-reversed scatter of the last
// pass would otherwise race with other workgroups still reading).
// prm = [lm, s, log_n, bit_reverse_output, scale_flag, scale (8 words, Montgomery form)]

const TILE_LOG: u32 = 9u;
const WG: u32 = 256u;

@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;        // 2 x vec4 per element
@group(0) @binding(1) var<storage, read> tw: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> prm: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<u32>>;
var<workgroup> tile: array<vec4<u32>, 1024>; // 512 elements x 32 bytes = 16 KB

fn load_data(i: u32) -> Fr { return fr_load(src[2u * i], src[2u * i + 1u]); }
fn load_tile(i: u32) -> Fr { return fr_load(tile[2u * i], tile[2u * i + 1u]); }
fn load_tw(i: u32) -> Fr { return fr_load(tw[2u * i], tw[2u * i + 1u]); }
fn store_data(i: u32, v: Fr) { let w = fr_words(v); dst[2u * i] = w.lo; dst[2u * i + 1u] = w.hi; }
fn store_tile(i: u32, v: Fr) { let w = fr_words(v); tile[2u * i] = w.lo; tile[2u * i + 1u] = w.hi; }

// Global index of tile slot `slot` for workgroup `wg`.
fn global_index(wg: u32, slot: u32, lm: u32, s: u32) -> u32 {
  let lg = TILE_LOG - s;                  // columns per workgroup = 2^lg
  let row = slot >> lg;
  let col = slot & ((1u << lg) - 1u);
  let c = (wg << lg) + col;               // column id over the whole array
  let j0 = c & ((1u << lm) - 1u);         // offset inside the stride
  return ((c >> lm) << (lm + s)) + j0 + (row << lm);
}

@compute @workgroup_size(256)
fn ntt_pass(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let lm = prm[0]; let s = prm[1]; let k = prm[2];
  let lg = TILE_LOG - s;
  let wg = wid.x; let t = lid.x;

  for (var e = 0u; e < 2u; e++) {
    let slot = t + e * WG;
    store_tile(slot, load_data(global_index(wg, slot, lm, s)));
  }
  workgroupBarrier();

  // One butterfly per thread per stage (512 / 2 = 256 butterflies).
  let col = t & ((1u << lg) - 1u);
  let q = t >> lg;
  let j0 = ((wg << lg) + col) & ((1u << lm) - 1u);
  for (var st = 0u; st < s; st++) {
    let lh = s - 1u - st;                         // log2 of row distance
    let r_lo = q & ((1u << lh) - 1u);
    let row_a = ((q >> lh) << (lh + 1u)) + r_lo;
    let slot_a = (row_a << lg) + col;
    let slot_b = slot_a + (1u << (lh + lg));
    let twiddle = load_tw(((r_lo << lm) + j0) << (k - 1u - lh - lm));
    let a = load_tile(slot_a);
    let b = load_tile(slot_b);
    store_tile(slot_a, fr_add(a, b));
    store_tile(slot_b, fr_mul(fr_sub(a, b), twiddle));
    workgroupBarrier();
  }

  for (var e = 0u; e < 2u; e++) {
    let slot = t + e * WG;
    var v = load_tile(slot);
    if (prm[4] == 1u) {
      v = fr_mul(v, fr_unpack(array<u32, 8>(prm[5], prm[6], prm[7], prm[8], prm[9], prm[10], prm[11], prm[12])));
    }
    var g = global_index(wg, slot, lm, s);
    if (prm[3] == 1u) { g = reverseBits(g) >> (32u - k); }
    store_data(g, v);
  }
}
