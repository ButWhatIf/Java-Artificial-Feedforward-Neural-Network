import java.util.function.Function;

/**
 * Provides a way to store numerical data in a two-dimensional array and perform elementary linear algebra
 * calculations such as multiplication and addition. For convenient programming, all indices start at 0.
 * Whenever possible, we fall upon the raw two-dimensional array in methods. This class forms the backbone
 * of most computations done in this program.
 */
public final class Matrix {
    private int rows;
    private int columns;
    // Matrix two-dimensional array representation. Stores the data. First index denotes rows, second columns
    private double[][] elementData;

    /**
     * Default constructor if needed
     */
    public Matrix() {
    }

    /**
     * Constructs the (rows) x (columns) zero matrix. Initiating as the zero matrix ends up being convenient
     * in many cases, especially when finding the sum of an array of matrices.
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
     * Constructs a matrix out of a given two-dimensional array.
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

    public double get(int rowIndex, int columnIndex) {
        return elementData[rowIndex][columnIndex];
    }

    /**
     * Multiplies two matrices. Order of input matters. Throws exception if the column count of the first matrix
     * is not the same as the column count of the second, as per the theorems of linear algebra.
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
     * Multiplies a matrix by a scalar.
     *
     * @param scalar The number by which to multiply the matrix
     * @param matr   The matrix to be multiplied
     * @return Matrix "matr" scalar multiplied by "scalar"
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
     * Finds the sum of two matrices with equal dimensions.
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
     * Used as a test matrix.
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
     * commutative. The Hadamard product is the element-wise multiplication of two matrices with equal dimensions.
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
     * Computes the Hadamard division of two matrices.
     * The Hadamard division is the element-wise division of two matrices with equal dimensions.
     *
     * @param matr1 Numerator matrix
     * @param matr2 Denominator matrix
     * @return Hadamard division
     */
    public static Matrix hDivide(Matrix matr1, Matrix matr2) {
        if (matr1.rows != matr2.rows || matr1.columns != matr2.columns) {
            throw new DataMismatchException("Matrices must be same dimension");
        } else {
            Matrix hadamard = new Matrix(matr1.rows, matr2.columns);
            for (int i = 0; i < matr1.rows; i++) {
                for (int j = 0; j < matr1.columns; j++) {
                    hadamard.elementData[i][j] = matr1.elementData[i][j] / matr2.elementData[i][j];
                }
            }
            return hadamard;
        }
    }

    /**
     * Takes the transpose (swaps rows and columns) of the matrix instance.
     *
     * @return Transposed matrix
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

    public Matrix forEach(Function<Double, Double> action) {
        Matrix newMatrix = new Matrix(rows, columns);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++) {
                newMatrix.elementData[i][j] = action.apply(this.elementData[i][j]);
            }
        }
        return newMatrix;
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
     * Returns the negative of the matrix. We decided to create a separate method for finding the negative matrix
     * because using the class .scale() method to find the negative was used often and the syntax was
     * cumbersome.
     *
     * @return The matrix scalar multiplied by -1.
     */
    public Matrix neg() {
        return this.forEach(i -> -i);
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
