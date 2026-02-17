# Modern Robotics in Rust

This repository contains Rust implementations of algorithms and concepts from the book "Modern Robotics: Mechanics, Planning, and Control" by Kevin M. Lynch and Frank C. Park.
All functions can be accessed via the root of this crate.
Short examples are provided in the documentation of each function.

## Notes
- I recommend that you only use the default formatter and not the debug formatter to print the matrices, because `nalgebra` uses a column major layout internally and therefore the debug format is transposed compared to the mathematical notation. The default formatter prints the matrices in the mathematical notation.
- Even though the code is not optimized it is still 10x to 100x faster than the python.
- There is a module with some helper macros like `mat4!` to create 4x4 matrices in a more readable way. You can also use the `nalgebra` functions to create matrices, but the macros are more concise for small matrices.

### Example - Inverse Kinematics

```rust
use modern_robotics_rust::mat4;
use modern_robotics_rust::{ikin_space, near_zero};
use nalgebra::SMatrix;
use nalgebra::Vector3;

fn main() {
    let s_list = SMatrix::<f64, 3, 6>::from_row_slice(&
        [0.0, 0.0, 1.0, 4.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, -1.0, -6.0, 0.0, -0.1,
        ]).transpose();
    let m = mat4!(-1.0, 0.0, 0.0, 0.0,
                   0.0, 1.0, 0.0, 6.0,
                   0.0, 0.0, -1.0, 2.0,
                   0.0, 0.0, 0.0, 1.0);
    let t = mat4!([0.0, 1.0, 0.0, -5.0],
                  [1.0, 0.0, 0.0, 4.0],
                  [0.0, 0.0, -1.0, 1.68584073],
                  [0.0, 0.0, 0.0, 1.0]);
    let thetalist0 = Vector3::new(1.5, 2.5, 3.0);
    let eomg = 0.01;
    let ev = 0.001;
    let (thetalist, success) = ikin_space(s_list, m, t, thetalist0, eomg, ev);
    println!("thetalist: {thetalist}");
    assert!(success);
    assert!(near_zero((thetalist - Vector3::new(1.57073783, 2.99966384, 3.1415342)).abs().max()));
}
```