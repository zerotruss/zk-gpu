// twiddle[i] = hi[i >> lo_bits] * lo[i & (2^lo_bits - 1)]   (all in Montgomery form)
@group(0) @binding(0) var<storage, read> hi: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> lo: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> prm: array<u32>; // [lo_bits, count]

fn rd(buf_hi: bool, i: u32) -> Fr {
  var a: vec4<u32>; var b: vec4<u32>;
  if (buf_hi) { a = hi[2u * i]; b = hi[2u * i + 1u]; } else { a = lo[2u * i]; b = lo[2u * i + 1u]; }
  return fr_unpack(array<u32, 8>(a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w));
}

@compute @workgroup_size(64)
fn twiddles(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= prm[1]) { return; }
  let lb = prm[0];
  let w = fr_pack(fr_mul(rd(true, i >> lb), rd(false, i & ((1u << lb) - 1u))));
  out[2u * i] = vec4<u32>(w[0], w[1], w[2], w[3]);
  out[2u * i + 1u] = vec4<u32>(w[4], w[5], w[6], w[7]);
}
