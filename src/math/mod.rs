use std::fmt::Display;

mod algebra;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of the sigmoid function.
pub fn sigmoid1(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

#[derive(Clone)]
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
    pub fn randomize(&self) -> Self {
        let mut result = self.clone();
        result.randomize_mut();
        result
    }
    pub fn randomize_mut(&mut self) -> &mut Self {
        for i in 0..self.0.len() {
            self.0[i] = rand::random::<f64>();
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
}

impl Iterator for Vector {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop()
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
                self.set(col, row, rand::random::<f64>());
            }
        }
        self
    }
}
