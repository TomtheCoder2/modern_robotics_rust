use nalgebra::{SMatrix, Vector3, RowVector3};
use modern_robotics_rust::utils::{block2x2, c_stack, r_stack};
use modern_robotics_rust::{r_, utils as fs};
use modern_robotics_rust::{c_, m3, m4};


#[test]
fn c_stack_2x_examples() {
    let a: SMatrix<f64, 2, 2> = SMatrix::from_row_slice(&[1., 2., 3., 4.]);
    let b: SMatrix<f64, 2, 1> = SMatrix::from_row_slice(&[5., 6.]);
    let c = c_stack(&a, &b); // 2x3
    // Should be [[1,2,5],[3,4,6]]
    assert_eq!(c[(0, 0)], 1.);
    assert_eq!(c[(0, 1)], 2.);
    assert_eq!(c[(0, 2)], 5.);
    assert_eq!(c[(1, 0)], 3.);
    assert_eq!(c[(1, 1)], 4.);
    assert_eq!(c[(1, 2)], 6.);

    let c2 = c_!(a, b);
    assert_eq!(c, c2);
}

#[test]
fn r_stack_2x_examples() {
    let a: SMatrix<f64, 2, 2> = SMatrix::from_row_slice(&[1., 2., 3., 4.]);
    let b: SMatrix<f64, 1, 2> = SMatrix::from_row_slice(&[5., 6.]);
    let r = r_stack(&a, &b); // 3x2
    // [[1,2],[3,4],[5,6]]
    assert_eq!(r[(0, 0)], 1.);
    assert_eq!(r[(0, 1)], 2.);
    assert_eq!(r[(1, 0)], 3.);
    assert_eq!(r[(1, 1)], 4.);
    assert_eq!(r[(2, 0)], 5.);
    assert_eq!(r[(2, 1)], 6.);

    let r2 = r_!(a, b);
    assert_eq!(r, r2);
}

#[test]
fn block2x2_works_3x3() {
    // Build a 3x3 from blocks: A:2x2, B:2x1, C:1x2, D:1x1
    let A: SMatrix<f64, 2, 2> = SMatrix::from_row_slice(&[1., 2., 3., 4.]);
    let B: SMatrix<f64, 2, 1> = SMatrix::from_row_slice(&[5., 6.]);
    let C: SMatrix<f64, 1, 2> = SMatrix::from_row_slice(&[7., 8.]);
    let D: SMatrix<f64, 1, 1> = SMatrix::from_row_slice(&[9.]);

    let M = block2x2(&A, &B, &C, &D);
    // Expect:
    // [1 2 5
    //  3 4 6
    //  7 8 9]
    let expect = SMatrix::<f64, 3, 3>::from_row_slice(&[1., 2., 5., 3., 4., 6., 7., 8., 9.]);
    assert_eq!(M, expect);
}

#[test]
fn literals_3x3_and_4x4() {
    let m3 = m3!(
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
    );
    // Expect visual row-major access
    assert_eq!(m3[(0, 0)], 1.);
    assert_eq!(m3[(0, 1)], 2.);
    assert_eq!(m3[(1, 2)], 6.);
    assert_eq!(m3[(2, 1)], 8.);

    let m4 = m4!(
        [1.,  2.,  3.,  4.],
        [5.,  6.,  7.,  8.],
        [9., 10., 11., 12.],
        [13.,14., 15., 16.],
    );
    assert_eq!(m4[(0, 0)], 1.);
    assert_eq!(m4[(0, 3)], 4.);
    assert_eq!(m4[(3, 0)], 13.);
    assert_eq!(m4[(3, 3)], 16.);
}

#[test]
fn small_vectors_helpers() {
    let r = fs::row3(1., 2., 3.);
    let c = fs::col3(1., 2., 3.);
    assert_eq!(r[(0, 2)], 3.);
    assert_eq!(c[(2, 0)], 3.);

    let r4 = fs::row4(1., 2., 3., 4.);
    let c4 = fs::col4(1., 2., 3., 4.);
    assert_eq!(r4[(0, 3)], 4.);
    assert_eq!(c4[(3, 0)], 4.);
}