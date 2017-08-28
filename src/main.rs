// monte carlo radiative transfer code based on plane-parallel approximation
//   It calculates radiative fluxes at top and bottom of the atmosphere.
//   It assumes the atmosphere as a single-layer plane-parallel atmosphere.

extern crate rand;

use rand::Rng;
use std::f64;
use std::ops;
use std::env;

const PI: f64 = 3.141592653589793238462643383279502884197;
const DTOR: f64 = PI / 180.0;
const DZ: f64 = 10000.0;
const NSMAX: u64 = 10000;
const TINY: f64 = 0.00000000001;

struct MCInput {
    tau: f64,
    omg: f64,
    asy: f64,
    alb: f64,
    the0: f64,
    nsmpl: u64,
}

struct MCOutput {
    flxr: f64,
    flxt: f64,
    flxa: f64,
    flxt0: f64,
    flxtd: f64,
}

#[derive(Copy)]
struct Vec3D {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3D {
    fn new(x: f64, y: f64, z: f64) -> Vec3D {
        Vec3D {
            x: x,
            y: y,
            z: z,
        }
    }
    fn new_with_cartesian(r: f64, the: f64, phi: f64) -> Vec3D {
        let cosq = the.cos();
        let sinq = the.sin();
        Vec3D {
            x: r * sinq * phi.cos(),
            y: r * sinq * phi.sin(),
            z: r * cosq,
        }
    }
    fn unit(&self) -> Vec3D {
        let norm = (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt();
        Vec3D {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }
    fn lambert_reflect(xi: f64, sinf: f64, cosf: f64) -> Vec3D {
        let cosq = xi.sqrt();
        let sinq = (1.0 - xi).sqrt();
        Vec3D {
            x: sinq * cosf,
            y: sinq * sinf,
            z: cosq,
        }.unit()
    }
    fn rotate(&self, cosq: f64, sinq: f64, cosf: f64, sinf: f64) -> Vec3D {
        let sinq2 = self.x.powi(2) + self.y.powi(2);
        let x: f64;
        let y: f64;
        let z: f64;
        if sinq2 > TINY {
           let sinq0 = sinq2.sqrt();
           let cosf0 = self.x / sinq0;
           let sinf0 = self.y / sinq0;
           x = cosq * self.x + sinq * (cosf * self.z * cosf0 - sinf * sinf0);
           y = cosq * self.y + sinq * (cosf * self.z * sinf0 + sinf * cosf0);
           z = cosq * self.z - sinq * cosf * sinq0;
        } else if self.z > 0.0 {
           x = sinq * cosf;
           y = sinq * sinf;
           z = cosq;
        } else {
           x = -sinq * cosf;
           y = -sinq * sinf;
           z = -cosq;
        }
        Vec3D {
            x: x,
            y: y,
            z: z,
        }
    }
}

impl Clone for Vec3D {
    fn clone(&self) -> Self {
        Vec3D {
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
        }
    } 
}

impl ops::Add<Vec3D> for Vec3D {
    type Output = Vec3D;

    fn add(self, rhs: Vec3D) -> Vec3D {
        Vec3D {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl ops::Add<f64> for Vec3D {
    type Output = Vec3D;

    fn add(self, rhs: f64) -> Vec3D {
        Vec3D {
            x: self.x + rhs,
            y: self.y + rhs,
            z: self.z + rhs,
        }
    }
}

impl ops::Mul<Vec3D> for Vec3D {
    type Output = Vec3D;

    fn mul(self, rhs: Vec3D) -> Vec3D {
        Vec3D {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl ops::Mul<f64> for Vec3D {
    type Output = Vec3D;

    fn mul(self, rhs: f64) -> Vec3D {
        Vec3D {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

fn rnd_uniform() -> f64 {
    rand::thread_rng().gen_range(0.0, 1.0)
}

fn rand_scat_hg(g: f64, xi: f64) -> (f64, f64) {
    let xi1 = 1.0 - 2.0 * xi;
    let gg = if g.abs() < TINY { TINY * g / g.abs() } else { g };
    let cosq = ((1.0 + gg * gg - ((1.0 - gg * gg) / (1.0 - gg * xi1)).powi(2)) / (2.0 * gg)).min(1.0).max(-1.0);
    let sinq = (1.0 - cosq * cosq).sqrt();
    (cosq, sinq)
}

fn mcpp_world(mc_input: MCInput) -> MCOutput {
    let tau = mc_input.tau;
    let omg = mc_input.omg;
    let asy = mc_input.asy;
    let alb = mc_input.alb;
    let the0 = mc_input.the0;
    let nsmpl = mc_input.nsmpl;

    let mut nph = 0;
    let mut nss = 0;
    let ext = tau / DZ;
    let mut sumr = 0.0;
    let mut sumt0 = 0.0;
    let mut sumtd = 0.0;

    let dir0 = Vec3D::new_with_cartesian(1.0, the0 * DTOR, 0.0) * (-1.0);

    loop {
        nph += 1;
        let mut w = 1.0;
        let mut dir = dir0;
        let mut loc = Vec3D::new(0.0, 0.0, DZ);

        for i in 0..NSMAX {
            let ftau = -f64::ln(f64::max(rnd_uniform(), TINY));
            let mut locnew = loc + dir * (ftau / ext);
            nss += 1;

            if locnew.z >= DZ {
                sumr += w;
                break;
            } else if locnew.z <= 0.0 {
                if i == 0 {
                    sumt0 += w;
                } else {
                    sumtd += w;
                }
                locnew = Vec3D {z: 0.0, .. locnew};
                w *= alb;
                let phi = 2.0 * PI * rnd_uniform();
                let cosf = phi.cos();
                let sinf = phi.sin();
                dir = Vec3D::lambert_reflect(rnd_uniform(), cosf, sinf);
            } else {
                w *= omg;
                let phi = 2.0 * PI * rnd_uniform();
                let cosf = phi.cos();
                let sinf = phi.sin();
                let (cosq, sinq) = rand_scat_hg(asy, rnd_uniform());
                dir = dir.rotate(cosq, sinq, cosf, sinf);
            }

            if w <= TINY { break; }
            loc = locnew;
        }

        if nss >= nsmpl { break; }
    }

    let flxr = sumr / (nph as f64);
    let flxt0 = sumt0 / (nph as f64);
    let flxtd = sumtd / (nph as f64);
    let flxt = flxt0 + flxtd;
    let flxa = 1.0 - flxr + flxt * (alb - 1.0);

    MCOutput {
        flxr: flxr,
        flxt: flxt,
        flxa: flxa,
        flxt0: flxt0,
        flxtd: flxtd,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let mc_input = MCInput {
        tau: args[1].trim().parse().expect("Please input correct value: tau"),
        omg: args[2].trim().parse().expect("Please input correct value: omg"),
        asy: args[3].trim().parse().expect("Please input correct value: asy"),
        alb: args[4].trim().parse().expect("Please input correct value: alb"),
        the0: args[5].trim().parse().expect("Please input correct value: the0"),
        nsmpl: args[6].trim().parse().expect("Please input correct value: nsmpl"),
    };

    println!("tau: {}", mc_input.tau);
    println!("omg: {}", mc_input.omg);
    println!("asy: {}", mc_input.asy);
    println!("alb: {}", mc_input.alb);
    println!("the0: {}", mc_input.the0);
    println!("nsmpl: {}", mc_input.nsmpl);
    println!("---");

    let output = mcpp_world(mc_input);

    println!("flxr: {}", output.flxr);
    println!("flxt: {}", output.flxt);
    println!("flxa: {}", output.flxa);
    println!("flxt0: {}", output.flxt0);
    println!("flxtd: {}", output.flxtd);
}
