// Rescale coordinates: mul(x, scale) = x * scale / R. With scale = R^2 mod p this turns plain
// coordinates into Montgomery form (x * R). prm = [count, scale as 8 words]; `points` holds `count` coordinates of 8 words each.
@group(0) @binding(0) var<storage, read_write> points: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> prm: array<u32>; // [count, r2 words...]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= prm[0]) { return; }
  let r2 = fq_unpack(array<u32, 8>(prm[1], prm[2], prm[3], prm[4], prm[5], prm[6], prm[7], prm[8]));
  let w = fq_words(fq_mul(fq_load(points[2u * i], points[2u * i + 1u]), r2));
  points[2u * i] = w.lo;
  points[2u * i + 1u] = w.hi;
}
