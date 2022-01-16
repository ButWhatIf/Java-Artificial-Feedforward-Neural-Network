
/**
 * This abstract class currently supports two optimizers.
 */
abstract class Optimizer {
    abstract Matrix optimize(Matrix weight, Matrix[] derivatives);
}

/**
 * Stochastic gradient descent optimizer. Even though titled a "stochastic gradient descent" optimizer,
 * can also support mini-batch and batch gradient descents. It is best practice to keep the learning rate small,
 * about 0.001 to 0.0001.
 */
class SGD extends Optimizer {
    private final double eta;

    public SGD(double eta) {
        this.eta = eta;
    }

    public Matrix optimize(Matrix toUpdate, Matrix[] derivatives) {
        Matrix sum = new Matrix(toUpdate.rows(), toUpdate.columns());
        for (Matrix derivative : derivatives) {
            sum = Matrix.sum(sum, derivative);
        }
        return Matrix.sum(toUpdate, Matrix.scale(-eta / derivatives.length, sum));
    }

}


/**
 * Stochastic, mini-batch, or batch gradient descent with momentum. Typically converges in fewer epochs than a standard
 * SGD optimizer, but at the expense of computation time and an additional hyper-parameter. It typically suffices
 * for the momentum to be about 0.90, though some tuning might be required.
 */
class Momentum extends Optimizer {
    private final double eta;
    private final double beta;

    /**
     * Stochastic, mini-batch, or batch gradient descent with momentum. Typically converges in fewer epochs than a
     * standard SGD optimizer, but at the expense of computation time and an additional hyper-parameter.
     * It typically suffices for the momentum to be about 0.90, though some tuning might be required.
     *
     * @param eta  Learning rate
     * @param beta Momentum
     */
    public Momentum(double eta, double beta) {
        this.eta = eta;
        this.beta = beta;
    }

    public Matrix optimize(Matrix toUpdate, Matrix[] derivatives) {
        Matrix sum = new Matrix(toUpdate.rows(), toUpdate.columns());
        for (Matrix derivative : derivatives) {
            sum = Matrix.sum(sum, derivative);
        }
        Matrix m = new Matrix(toUpdate.rows(), toUpdate.columns());
        m = Matrix.sum(Matrix.scale(beta, m), Matrix.scale((-1.0) * eta / derivatives.length, sum));
        return Matrix.sum(toUpdate, m);
    }
}




