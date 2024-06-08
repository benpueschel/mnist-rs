use super::{Matrix, Vector};
use std::ops::*;

impl Vector {
    pub fn map(mut self, func: fn(f64) -> f64) -> Self {
        for i in 0..self.0.len() {
            self.0[i] = func(self.0[i]);
        }
        self
    }

}

impl Div<f64> for Vector {
    type Output = Self;

    fn div(mut self, scalar: f64) -> Self {
        self /= scalar;
        self
    }
}

impl DivAssign<f64> for Vector {
    fn div_assign(&mut self, scalar: f64) {
        for i in 0..self.0.len() {
            self.0[i] /= scalar;
        }
    }
}

impl Add<&Vector> for Vector {
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self += other;
        self
    }
}

impl Add for Vector {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, other: Self) {
        for i in 0..self.0.len() {
            self.0[i] += other.0[i];
        }
    }
}

impl AddAssign<&Vector> for Vector {
    fn add_assign(&mut self, other: &Self) {
        for i in 0..self.0.len() {
            self.0[i] += other.0[i];
        }
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        self -= other;
        self
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, other: Self) {
        for i in 0..self.0.len() {
            self.0[i] -= other.0[i];
        }
    }
}

impl SubAssign<&Vector> for Vector {
    fn sub_assign(&mut self, other: &Self) {
        for i in 0..self.0.len() {
            self.0[i] -= other.0[i];
        }
    }
}

impl Mul<f64> for Vector {
    type Output = Self;

    fn mul(mut self, scalar: f64) -> Self {
        self *= scalar;
        self
    }
}

impl MulAssign<f64> for Vector {
    fn mul_assign(&mut self, scalar: f64) {
        for i in 0..self.0.len() {
            self.0[i] *= scalar;
        }
    }
}

impl Mul<Vector> for f64 {
    type Output = Vector;

    fn mul(self, mut vec: Vector) -> Vector {
        vec *= self;
        vec
    }
}

// Matrix

impl Add for Matrix {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, other: Self) {
        assert_eq!(
            self._cols, other._cols,
            "Matrix dimensions do not match. Self columns: {}, other columns: {}",
            self._cols, other._cols
        );
        assert_eq!(
            self._rows, other._rows,
            "Matrix dimensions do not match. Self rows: {}, other rows: {}",
            self._rows, other._rows
        );
        for col in 0..self._cols {
            for row in 0..self._rows {
                self.set(col, row, self.at(col, row) + other.at(col, row));
            }
        }
    }
}

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, other: &Self) {
        assert_eq!(
            self._cols, other._cols,
            "Matrix dimensions do not match. Self columns: {}, other columns: {}",
            self._cols, other._cols
        );
        assert_eq!(
            self._rows, other._rows,
            "Matrix dimensions do not match. Self rows: {}, other rows: {}",
            self._rows, other._rows
        );
        for col in 0..self._cols {
            for row in 0..self._rows {
                self.set(col, row, self.at(col, row) + other.at(col, row));
            }
        }
    }
}

impl Sub for Matrix {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self {
        self -= other;
        self
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, other: Self) {
        assert_eq!(
            self._cols, other._cols,
            "Matrix dimensions do not match. Self columns: {}, other columns: {}",
            self._cols, other._cols
        );
        assert_eq!(
            self._rows, other._rows,
            "Matrix dimensions do not match. Self rows: {}, other rows: {}",
            self._rows, other._rows
        );
        for col in 0..self._cols {
            for row in 0..self._rows {
                self.set(col, row, self.at(col, row) - other.at(col, row));
            }
        }
    }
}

impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, other: &Self) {
        assert_eq!(
            self._cols, other._cols,
            "Matrix dimensions do not match. Self columns: {}, other columns: {}",
            self._cols, other._cols
        );
        assert_eq!(
            self._rows, other._rows,
            "Matrix dimensions do not match. Self rows: {}, other rows: {}",
            self._rows, other._rows
        );
        for col in 0..self._cols {
            for row in 0..self._rows {
                self.set(col, row, self.at(col, row) - other.at(col, row));
            }
        }
    }
}

impl Div<f64> for Matrix {
    type Output = Self;

    fn div(mut self, scalar: f64) -> Self {
        self /= scalar;
        self
    }
}

impl DivAssign<f64> for &mut Matrix {
    fn div_assign(&mut self, scalar: f64) {
        for col in 0..self._cols {
            for row in 0..self._rows {
                self.set(col, row, self.at(col, row) / scalar);
            }
        }
    }
}

impl DivAssign<f64> for Matrix {
    fn div_assign(&mut self, scalar: f64) {
        for col in 0..self._cols {
            for row in 0..self._rows {
                self.set(col, row, self.at(col, row) / scalar);
            }
        }
    }
}

impl Mul<f64> for Matrix {
    type Output = Self;

    fn mul(mut self, scalar: f64) -> Self {
        self *= scalar;
        self
    }
}

impl MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, scalar: f64) {
        for col in 0..self._cols {
            for row in 0..self._rows {
                self.set(col, row, self.at(col, row) * scalar);
            }
        }
    }
}

impl Mul<Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, mut mat: Matrix) -> Matrix {
        mat *= self;
        mat
    }
}

impl Mul<&Vector> for Matrix {
    type Output = Vector;

    fn mul(self, vec: &Vector) -> Vector {
        &self * vec
    }
}

impl Mul<Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, vec: Vector) -> Vector {
        self * &vec
    }
}

impl Mul<Vector> for Matrix {
    type Output = Vector;

    fn mul(self, vec: Vector) -> Vector {
        &self * &vec
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, vec: &Vector) -> Vector {
        let matrix_cols = self._cols;
        let vec_rows = vec.0.len();

        assert_eq!(
            matrix_cols, vec_rows,
            "Matrix and vector dimensions do not match. Matrix columns: {}, vector rows: {}",
            vec_rows, matrix_cols
        );

        let mut result = Vector::new(self._rows);
        for i in 0..self._rows {
            for j in 0..self._cols {
                result.0[i] += self.at(j, i) * vec.0[j];
            }
        }
        result
    }
}

