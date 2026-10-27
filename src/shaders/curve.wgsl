// BN254 G1 (y^2 = x^3 + 3) over Fe in Montgomery form.
//
// Accumulators use "XYZZ" coordinates: x = X/ZZ, y = Y/ZZZ with ZZ^3 = ZZZ^2. They make
// mixed addition (accumulator + affine point) cost 8M + 2S, the hot operation of Pippenger.
// Infinity is ZZ = 0 in XYZZ and (0, 0) in affine form (not on the curve).

struct Affine { x: Fe, y: Fe }
struct Xyzz { x: Fe, y: Fe, zz: Fe, zzz: Fe }

fn fe_dbl(a: Fe) -> Fe { return fe_add(a, a); }

fn xyzz_infinity() -> Xyzz { return Xyzz(FE_ZERO, FE_ZERO, FE_ZERO, FE_ZERO); }
fn xyzz_is_infinity(p: Xyzz) -> bool { return fe_is_zero(p.zz); }
fn affine_is_infinity(p: Affine) -> bool { return fe_is_zero(p.x) && fe_is_zero(p.y); }
fn xyzz_from_affine(p: Affine) -> Xyzz {
  if (affine_is_infinity(p)) { return xyzz_infinity(); }
  return Xyzz(p.x, p.y, FE_ONE, FE_ONE);
}
fn affine_neg(p: Affine) -> Affine { return Affine(p.x, fe_sub(FE_ZERO, p.y)); }
fn xyzz_neg(p: Xyzz) -> Xyzz { return Xyzz(p.x, fe_sub(FE_ZERO, p.y), p.zz, p.zzz); }

// 2P  (dbl-2008-s-1 with a = 0): 6M + 3S
fn xyzz_double(p: Xyzz) -> Xyzz {
  let u = fe_dbl(p.y);
  let v = fe_sqr(u);
  let w = fe_mul(u, v);
  let s = fe_mul(p.x, v);
  let x2 = fe_sqr(p.x);
  let m = fe_add(fe_dbl(x2), x2);
  let x3 = fe_sub(fe_sqr(m), fe_dbl(s));
  let y3 = fe_sub(fe_mul(m, fe_sub(s, x3)), fe_mul(w, p.y));
  return Xyzz(x3, y3, fe_mul(v, p.zz), fe_mul(w, p.zzz));
}

// P + Q with Q affine (madd-2008-s): 8M + 2S. Handles every special case.
fn xyzz_add_affine(p: Xyzz, q: Affine) -> Xyzz {
  if (affine_is_infinity(q)) { return p; }
  if (xyzz_is_infinity(p)) { return xyzz_from_affine(q); }
  let u2 = fe_mul(q.x, p.zz);
  let s2 = fe_mul(q.y, p.zzz);
  let pp = fe_sub(u2, p.x);
  let r = fe_sub(s2, p.y);
  if (fe_is_zero(pp)) {
    if (fe_is_zero(r)) { return xyzz_double(p); }
    return xyzz_infinity();
  }
  let ppsq = fe_sqr(pp);
  let ppp = fe_mul(pp, ppsq);
  let q_ = fe_mul(p.x, ppsq);
  let x3 = fe_sub(fe_sub(fe_sqr(r), ppp), fe_dbl(q_));
  let y3 = fe_sub(fe_mul(r, fe_sub(q_, x3)), fe_mul(p.y, ppp));
  return Xyzz(x3, y3, fe_mul(p.zz, ppsq), fe_mul(p.zzz, ppp));
}

// P + Q, both XYZZ (add-2008-s): 12M + 2S.
fn xyzz_add(p: Xyzz, q: Xyzz) -> Xyzz {
  if (xyzz_is_infinity(q)) { return p; }
  if (xyzz_is_infinity(p)) { return q; }
  let u1 = fe_mul(p.x, q.zz);
  let u2 = fe_mul(q.x, p.zz);
  let s1 = fe_mul(p.y, q.zzz);
  let s2 = fe_mul(q.y, p.zzz);
  let pp = fe_sub(u2, u1);
  let r = fe_sub(s2, s1);
  if (fe_is_zero(pp)) {
    if (fe_is_zero(r)) { return xyzz_double(p); }
    return xyzz_infinity();
  }
  let ppsq = fe_sqr(pp);
  let ppp = fe_mul(pp, ppsq);
  let q_ = fe_mul(u1, ppsq);
  let x3 = fe_sub(fe_sub(fe_sqr(r), ppp), fe_dbl(q_));
  let y3 = fe_sub(fe_mul(r, fe_sub(q_, x3)), fe_mul(s1, ppp));
  return Xyzz(x3, y3, fe_mul(fe_mul(p.zz, q.zz), ppsq), fe_mul(fe_mul(p.zzz, q.zzz), ppp));
}

// Memory format: an Affine is 2 field elements, an Xyzz is 4 (x, y, zz, zzz), FW vec4 each.
const AW: u32 = 2u * FW;
const XW: u32 = 4u * FW;

fn xyzz_pack(p: Xyzz) -> array<vec4<u32>, XW> {
  let x = fe_pack(p.x); let y = fe_pack(p.y); let zz = fe_pack(p.zz); let zzz = fe_pack(p.zzz);
  var out: array<vec4<u32>, XW>;
  for (var k = 0u; k < FW; k++) { out[k] = x[k]; out[FW + k] = y[k]; out[2u * FW + k] = zz[k]; out[3u * FW + k] = zzz[k]; }
  return out;
}
fn xyzz_unpack(a: array<vec4<u32>, XW>) -> Xyzz {
  var x: array<vec4<u32>, FW>; var y: array<vec4<u32>, FW>; var zz: array<vec4<u32>, FW>; var zzz: array<vec4<u32>, FW>;
  for (var k = 0u; k < FW; k++) { x[k] = a[k]; y[k] = a[FW + k]; zz[k] = a[2u * FW + k]; zzz[k] = a[3u * FW + k]; }
  return Xyzz(fe_unpack(x), fe_unpack(y), fe_unpack(zz), fe_unpack(zzz));
}
fn affine_unpack(a: array<vec4<u32>, AW>) -> Affine {
  var x: array<vec4<u32>, FW>; var y: array<vec4<u32>, FW>;
  for (var k = 0u; k < FW; k++) { x[k] = a[k]; y[k] = a[FW + k]; }
  return Affine(fe_unpack(x), fe_unpack(y));
}
