// Tree-sum step: every thread adds up to G consecutive items of one window.
// prm = [len_in]; out has ceil(len_in / G) items per window. Constants W, G are prepended.

@group(0) @binding(0) var<storage, read> input: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> prm: array<u32>;

fn load_item(i: u32) -> Xyzz {
  var a: array<vec4<u32>, XW>;
  for (var k = 0u; k < XW; k++) { a[k] = input[XW * i + k]; }
  return xyzz_unpack(a);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let len = prm[0];
  let out_len = (len + G - 1u) / G;
  let t = id.x;
  if (t >= W * out_len) { return; }
  let start = (t / out_len) * len + (t % out_len) * G;
  let end = min(G, len - (t % out_len) * G);
  var acc = xyzz_infinity();
  for (var k = 0u; k < end; k++) { acc = xyzz_add(acc, load_item(start + k)); }
  let a = xyzz_pack(acc);
  for (var k = 0u; k < XW; k++) { out[XW * t + k] = a[k]; }
}
