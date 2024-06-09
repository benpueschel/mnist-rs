use std::{fmt::Display, ops::{Index, IndexMut}};

pub mod algebra;

pub fn softmax(mut x: Vector) -> Vector {
    let d = -x[x.argmax()];
    for i in 0..x.0.len() {
        x[i] = (x[i] + d).exp();
    }
    let sum = x.sum_values();
    x / sum
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    _rows: usize,
    _cols: usize,
    _data: Vec<Vector>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vector(pub Vec<f64>);

impl From<Vec<f64>> for Vector {
    fn from(data: Vec<f64>) -> Self {
        Vector(data)
    }
}

impl Vector {
    pub fn new(len: usize) -> Vector {
        Vector(vec![0.0; len])
    }
    pub fn set(&mut self, index: usize, value: f64) {
        self.0[index] = value;
    }
    pub fn sum_values(&self) -> f64 {
        self.0.iter().sum()
    }

    pub fn randomize(&self) -> Self {
        let mut result = self.clone();
        result.randomize_mut();
        result
    }
    pub fn randomize_mut(&mut self) -> &mut Self {
        for i in 0..self.0.len() {
            self.0[i] = rand::random::<f64>() - 0.5;
        }
        self
    }

    pub fn argmax(&self) -> usize {
        let mut max = 0;
        for i in 1..self.0.len() {
            if self.0[i] > self.0[max] {
                max = i;
            }
        }
        max
    }

    pub fn at(&self, index: usize) -> f64 {
        self.0[index]
    }
}

impl Iterator for Vector {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
    }

    fn max(self) -> Option<Self::Item> {
        self.0.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap())
    }
    fn sum<S>(self) -> S
        where
            Self: Sized,
            S: std::iter::Sum<Self::Item>, {
        self.0.into_iter().sum()
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix({}x{})", self._rows, self._cols)
    }
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        Matrix {
            _rows: rows,
            _cols: columns,
            _data: vec![Vector::new(rows); columns],
        }
    }
    pub fn rows(&self) -> usize {
        self._rows
    }
    pub fn cols(&self) -> usize {
        self._cols
    }
    pub fn data(&self) -> &[Vector] {
        &self._data
    }
    pub fn at(&self, col: usize, row: usize) -> f64 {
        self._data[col].0[row]
    }
    pub fn vector_at(&self, col: usize) -> &Vector {
        &self._data[col]
    }

    pub fn set(&mut self, col: usize, row: usize, data: f64) {
        self._data[col].0[row] = data;
    }
    pub fn randomize(&self) -> Self {
        let mut result = self.clone();
        result.randomize_mut();
        result
    }
    pub fn randomize_mut(&mut self) -> &mut Self {
        for col in 0..self._cols {
            for row in 0..self._rows {
                self.set(col, row, rand::random::<f64>() - 0.5);
            }
        }
        self
    }
}

impl Index<usize> for Matrix {
    type Output = Vector;

    fn index(&self, index: usize) -> &Self::Output {
        &self._data[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self._data[index]
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
