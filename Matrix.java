import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Provides a way to store numerical data in a two-dimensional array and perform elementary linear algebra
 * calculations such as multiplication and addition. For convenient programming, all indices start at 0.
 */
public final class Matrix {
    /*
     * Row count
     */
    private int rows;
    /*
     * Column count
     */
    private int columns;
    /*
     * Matrix two-dimensional array representation. Stores the data.
     */
    private double[][] elementData;


    public Matrix() {
    }

    /**
     * Constructs the (rows) x (columns) zero matrix
     *
     * @param rows    row count
     * @param columns column count
     */
    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.elementData = new double[this.rows][this.columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                elementData[i][j] = 0.0;
            }
        }
    }

    /**
     * Constructs a matrix out of a given two-dimensional array
     *
     * @param matrix Two dimensional array representation of a matrix. Each array contained in this two-dimensional
     *               array represents a row, with the index
     */
    public Matrix(double[][] matrix) {
        int colDim = matrix[0].length;
        for (double[] rows : matrix) {
            if (colDim != rows.length) {
                throw new DataMismatchException("All arrays must be same length");
            }
        }
        elementData = matrix;
        rows = matrix.length;
        columns = colDim;
    }

    public int[] size() {
        return new int[]{rows, columns};
    }

    @SuppressWarnings("unchecked")
    public double get(int rowIndex, int columnIndex) {
        return this.elementData[rowIndex][columnIndex];
    }

    /**
     * Multiplies two matrices. Order of input matters. Throws exception if the column count of the first matrix
     * is not the same as the column count of the second, as per the theorems of linear algebra. Computationally
     * expensive, should avoid calling unless absolutely necessary.
     *
     * @param matr1 Right matrix
     * @param matr2 Left matrix
     * @return The multiplied matrix. Has dimension of (matr1.rows) x (matr2.columns).
     */
    public static Matrix multiply(Matrix matr1, Matrix matr2) {
        if (matr1.columns != matr2.rows) {
            throw new DataMismatchException("First matrix column count and second matrix row count unequal");
        }
        Matrix product = new Matrix(matr1.rows, matr2.columns);
        for (int i = 0; i < matr1.rows; i++) {
            for (int j = 0; j < matr2.columns; j++) {
                for (int k = 0; k < matr1.columns; k++) {
                    product.elementData[i][j] = product.elementData[i][j] +
                            matr1.elementData[i][k] * matr2.elementData[k][j];
                }
            }
        }
        return product;
    }

    /**
     * Multiplies a matrix by a scalar. Each element is multiplied by the scalar
     *
     * @param scalar The number by which to multiply the matrix
     * @param matr   The matrix to be multiplied
     * @return
     */
    public static Matrix scale(double scalar, Matrix matr) {
        for (int i = 0; i < matr.rows; i++) {
            for (int j = 0; j < matr.columns; j++) {
                matr.elementData[i][j] = scalar * matr.elementData[i][j];
            }
        }
        return matr;
    }

    /**
     * Finds the sum of two matrices of equal dimension
     *
     * @param matr1 First matrix
     * @param matr2 Second matrix
     * @return Sum of the matrices
     */
    public static Matrix sum(Matrix matr1, Matrix matr2) {
        if (matr1.rows != matr2.rows || matr1.columns != matr2.columns) {
            throw new DataMismatchException("Matrices must have same dimension");
        }
        Matrix sum = new Matrix(matr1.rows, matr1.columns);
        for (int i = 0; i < matr1.rows; i++) {
            for (int j = 0; j < matr1.columns; j++) {
                sum.elementData[i][j] = matr1.elementData[i][j] + matr2.elementData[i][j];
            }
        }
        return sum;
    }

    /**
     * Returns an (dim) x (dim) identity matrix.
     *
     * @param dim Dimension of the matrix
     * @return (dim) x (dim) identity matrix
     */
    public static Matrix IDENTITY(int dim) {
        Matrix id = new Matrix(dim, dim);
        for (int i = 0; i < dim; i++) {
            id.elementData[i][i] = 1.0;
        }
        return id;
    }

    /**
     * Computes the Hadamard product of two matrices. Unlike the ordinary matrix product, the Hadamard product is
     * commutative.
     *
     * @param matr1 First matrix
     * @return Hadamard product
     */
    public static Matrix hMultiply(Matrix matr1, Matrix matr2) {
        if (matr1.rows != matr2.rows || matr1.columns != matr2.columns) {
            throw new DataMismatchException("Matrices must be same dimension");
        } else {
            Matrix hadamard = new Matrix(matr1.rows, matr2.columns);
            for (int i = 0; i < matr1.rows; i++) {
                for (int j = 0; j < matr1.columns; j++) {
                    hadamard.elementData[i][j] = matr1.elementData[i][j] * matr2.elementData[i][j];
                }
            }
            return hadamard;
        }
    }

    /**
     * Takes the transpose (swaps rows and columns) of the matrix instance.
     */
    public Matrix transpose() {
        Matrix transposed = new Matrix(columns, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                transposed.elementData[j][i] = elementData[i][j];
            }
        }
        return transposed;
    }

    /**
     * Clones the matrix and saves the clone into this instance of matrix
     *
     * @param newMatrix Matrix to clone
     */
    public void replaceWith(Matrix newMatrix) {
        this.elementData = newMatrix.elementData.clone();
        this.rows = newMatrix.rows;
        this.columns = newMatrix.columns;
    }

    /**
     * Adds a new row to the matrix. This new row will be the last row of the matrix. First entry will be in first
     * column, second entry in second column, etc.
     *
     * @param newRow Column to be added.
     */
    public void addRow(double[] newRow) {
        if (newRow.length != columns) {
            throw new DataMismatchException("Invalid array size: " + newRow.length + " columns received, " +
                    columns + " expected.");
        }
        double[][] temp = this.elementData;
        this.elementData = new double[++rows][columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (i < temp.length) {
                    this.elementData[i][j] = temp[i][j];
                } else {
                    this.elementData[i] = newRow;
                }
            }
        }
    }

    /**
     * Adds a new column to the end of the matrix, with the first entry populating the first row,
     * second entry populating the second row, etc.
     *
     * @param column Column to be added.
     */
    public void addColumn(double[] column) {
        if (column.length != rows) {
            throw new DataMismatchException("Invalid array size: " + column.length + " rows received, " + rows +
                    " expected.");
        }
        double[][] temp = this.elementData;
        this.elementData = new double[rows][++columns];
        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < columns; j++) {
                if (j < columns - 1) {
                    this.elementData[i][j] = temp[i][j];
                } else {
                    this.elementData[i][j] = column[i];
                }
            }
        }
    }

    /**
     * @return String representation of Matrix instance
     */
    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                str.append(this.elementData[i][j]).append(" ");
            }
            str.append("\n");
        }
        return str.toString();
    }

    @Deprecated
    public void forEach(Consumer<? super Number> action) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++) {
                action.accept(this.elementData[i][j]);
            }
        }
    }

    public void forEach(Function<Double, Double> action) {
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++) {
                elementData[i][j] = action.apply(this.elementData[i][j]);
            }
        }
    }

    public void setElement(int row, int column, Double value) {
        this.elementData[row][column] = value;
    }

    /**
     * Finds the number of rows of an instance of Matrix
     *
     * @return The number of rows of the matrix
     */
    public int rows() {
        return this.rows;
    }

    /**
     * Column count for this instance of Matrix
     *
     * @return The number of columns
     */
    public int columns() {
        return this.columns;
    }

    /**
     * Initializes each entry in the matrix to a random value between 0 and 1 exclusive and inclusive.
     */
    public void randomize() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                this.setElement(i, j, Math.random());
            }
        }
    }

    /**
     * Computes the norm of a vector (i.e., a matrix with only one column).
     *
     * @return The Euclidean norm of the vector.
     */
    public Double norm() {
        if (columns != 1) {
            throw new DataMismatchException("Cannot compute norm of this vector. Expected 1 column, received " +
                    columns);
        }

        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += Math.pow(elementData[i][0], 2);
        }

        return Math.sqrt(sum);
    }

    /**
     * Generic runtime error thrown if matrix operations cannot be performed with the provided arguments(s).
     */
    static class DataMismatchException extends RuntimeException {
        DataMismatchException(String message) {
            super(message);
        }
    }

}
