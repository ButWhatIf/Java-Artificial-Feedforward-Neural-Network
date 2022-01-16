import java.util.function.Function;

abstract class Cost {
    abstract double loss(Matrix actual, Matrix expected);

    abstract Matrix prime(Matrix actual, Matrix expected);
}

class MeanSquaredError extends Cost {
    public double loss(Matrix actual, Matrix expected) {
        Matrix difference = Matrix.sum(actual, expected.neg());
        return difference.norm() * difference.norm() / 2;
    }

    public Matrix prime(Matrix actual, Matrix expected) {
        return Matrix.sum(actual, expected.neg());
    }
}

class MeanAverageError extends Cost {
    public double loss(Matrix actual, Matrix expected) {
        Matrix diff = Matrix.sum(actual, expected.neg());
        return diff.norm() / 2;
    }

    public Matrix prime(Matrix actual, Matrix expected) {
        Matrix diff = Matrix.sum(actual, expected.neg());
        double scalar = diff.norm();
        return Matrix.scale(1.0 / scalar, diff.transpose()).transpose();
    }
}
