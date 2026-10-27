//! The two BN254 curve groups the MSM supports, and the WGSL that adapts the shared shaders
//! (`curve.wgsl` and the `msm_*.wgsl` passes) to each.
//!
//! G1 has coordinates in Fq. G2 has coordinates in Fq2 = `Fq[u] / (u^2 + 1)`, written `a + b*u`.
//! The group law (curve.wgsl) only needs a field element type `Fe` with a few operations, so
//! each group provides that and nothing else changes.

use crate::field;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Group {
    G1,
    G2,
}

const G1_PRELUDE: &str = "
const FW: u32 = 2u; // vec4 per field element
alias Fe = Fq;
const FE_ZERO = FQ_ZERO;
const FE_ONE = FQ_ONE;
fn fe_add(a: Fe, b: Fe) -> Fe { return fq_add(a, b); }
fn fe_sub(a: Fe, b: Fe) -> Fe { return fq_sub(a, b); }
fn fe_mul(a: Fe, b: Fe) -> Fe { return fq_mul(a, b); }
fn fe_sqr(a: Fe) -> Fe { return fq_mul(a, a); }
fn fe_is_zero(a: Fe) -> bool { return fq_is_zero(a); }
fn fe_unpack(a: array<vec4<u32>, FW>) -> Fe { return fq_load(a[0], a[1]); }
fn fe_pack(a: Fe) -> array<vec4<u32>, FW> { let w = fq_words(a); return array<vec4<u32>, FW>(w.lo, w.hi); }
";

// Fq2 with u^2 = -1: (a + bu)(c + du) = (ac - bd) + ((a + b)(c + d) - ac - bd)u  (3 Fq multiplications).
const G2_PRELUDE: &str = "
const FW: u32 = 4u;
struct Fe { a: Fq, b: Fq }
const FE_ZERO = Fe(FQ_ZERO, FQ_ZERO);
const FE_ONE = Fe(FQ_ONE, FQ_ZERO);
fn fe_add(x: Fe, y: Fe) -> Fe { return Fe(fq_add(x.a, y.a), fq_add(x.b, y.b)); }
fn fe_sub(x: Fe, y: Fe) -> Fe { return Fe(fq_sub(x.a, y.a), fq_sub(x.b, y.b)); }
fn fe_mul(x: Fe, y: Fe) -> Fe {
  let ac = fq_mul(x.a, y.a);
  let bd = fq_mul(x.b, y.b);
  let cross = fq_mul(fq_add(x.a, x.b), fq_add(y.a, y.b));
  return Fe(fq_sub(ac, bd), fq_sub(fq_sub(cross, ac), bd));
}
// (a + bu)^2 = (a + b)(a - b) + 2ab u  (2 Fq multiplications)
fn fe_sqr(x: Fe) -> Fe {
  let ab = fq_mul(x.a, x.b);
  return Fe(fq_mul(fq_add(x.a, x.b), fq_sub(x.a, x.b)), fq_add(ab, ab));
}
fn fe_is_zero(x: Fe) -> bool { return fq_is_zero(x.a) && fq_is_zero(x.b); }
fn fe_unpack(w: array<vec4<u32>, FW>) -> Fe { return Fe(fq_load(w[0], w[1]), fq_load(w[2], w[3])); }
fn fe_pack(x: Fe) -> array<vec4<u32>, FW> {
  let a = fq_words(x.a); let b = fq_words(x.b);
  return array<vec4<u32>, FW>(a.lo, a.hi, b.lo, b.hi);
}
";

impl Group {
    /// u32 words in the memory format of one affine point (x then y; Fq2 is a then b).
    pub fn affine_words(self) -> usize {
        match self {
            Group::G1 => 16,
            Group::G2 => 32,
        }
    }

    /// u32 words in the memory format of one XYZZ accumulator point.
    pub fn xyzz_words(self) -> usize {
        2 * self.affine_words()
    }

    /// Fq field code, the group's `Fe` prelude, and the group law: everything a point shader needs.
    pub fn wgsl(self) -> String {
        let prelude = match self {
            Group::G1 => G1_PRELUDE,
            Group::G2 => G2_PRELUDE,
        };
        format!("{}\n{prelude}\n{}", field::wgsl("fq", field::FQ), include_str!("shaders/curve.wgsl"))
    }
}
