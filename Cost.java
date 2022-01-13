import java.util.function.Function;

abstract class Cost {
    abstract double loss(Matrix actual, Matrix expected);

    abstract Matrix lossDerivative(Matrix actual, Matrix expected);
}

class LeastSquaresError extends Cost {
    public double loss(Matrix actual, Matrix expected) {
        Matrix difference = Matrix.sum(actual, Matrix.scale(-1,expected));
        return difference.norm() / 2;
    }

    public Matrix lossDerivative(Matrix actual, Matrix expected) {
        return Matrix.sum(actual, Matrix.scale(-1, expected));
    }

}
