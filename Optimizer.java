import java.util.ArrayList;
import java.util.HashMap;

/**
 * Gradient-descent based optimizers. Currently does not support Nesterov optimization variants.
 * <p>
 * Supports:
 * Ordinary Gradient Descent "SGD"
 * Momentum Gradient Descent "Momentum"
 * AdaGrad "AdaGrad"
 * RMSProp "RMSProp"
 * Adaptive Momentum Estimation "Adam"
 */
abstract class Optimizer {
    /**
     * Generic optimizer method. Moves the matrix "toUpdate" in the direction of the negative gradient.
     *
     * @param toUpdate
     * @param derivatives
     * @return
     */
    abstract Matrix optimize(Matrix toUpdate, Matrix[] derivatives);

    /**
     * Method for resetting the memory HashMaps
     */
    abstract void reset();
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

    /**
     * Not needed since SGD does not factor in previous gradient terms
     */
    @Deprecated
    public void reset() {}

}


/**
 * Stochastic, mini-batch, or batch gradient descent with momentum. Typically converges in fewer epochs than a standard
 * SGD optimizer, but at the expense of computation time and an additional hyper-parameter. It typically suffices
 * for the momentum to be about 0.90, though some tuning might be required.
 */
class Momentum extends Optimizer {
    private final double eta;
    private final double beta;
    // Stores previous gradients for each layer in this ArrayList for momentum. First index denotes
    private final HashMap<Matrix, Matrix> memory = new HashMap<>();

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

    /**
     * Momentum gradient descent algorithm. The HashMap instance "memory" stores the old values of the momentum matrix
     * for a given matrix. Since we always feed the output of this method back into itself in the next epoch, we can
     * set the key to access this old value of m as the output of the method itself. If the key does not exist
     * (i.e., running through the first epoch), we create the new key-value mapping. Once we no longer need the old
     * value of m, we dump the old key-value mapping from memory and replace it with the new. This algorithm enables
     * us to employ this descent method without losing abstraction.
     *
     * @param toUpdate    The matrix to update
     * @param derivatives The gradient of the matrix
     * @return The updated matrix.
     */
    public Matrix optimize(Matrix toUpdate, Matrix[] derivatives) {
        Matrix m;
        Matrix sum = new Matrix(toUpdate.rows(), toUpdate.columns());
        for (Matrix derivative : derivatives) { // Summing all the derivatives
            sum = Matrix.sum(sum, derivative);
        }
        if (!memory.containsKey(toUpdate)) {
            m = Matrix.scale(-eta / derivatives.length, sum);
        } else {
            m = memory.get(toUpdate);
            m = Matrix.sum(Matrix.scale(beta, m), Matrix.scale(-eta / derivatives.length, sum));
            memory.remove(toUpdate);
        }
        Matrix step = Matrix.sum(toUpdate, m);
        memory.put(step, m);
        return step;
    }

    public void reset() {
        memory.clear();
    }
}

/**
 * Adaptive Momentum Estimation optimizer. Stochastic, mini-batch, or batch gradient descent with momentum.
 * Typically converges in fewer epochs than a standard SGD optimizer, but at the expense of computation time and
 * 3 additional hyper-parameters.
 */
class ADAM extends Optimizer {
    private final double eta;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private int i = 0; // Iteration count
    // Stores previous gradients in these ArrayLists for momentum.
    private final HashMap<Matrix, Matrix> mMemory = new HashMap<>();
    private final HashMap<Matrix, Matrix> sMemory = new HashMap<>();


    /**
     * ADAM requires 4 positive hyper-parameters: learning rate, momentum decay rate, scaling decay rate, and a
     * smoothing term.
     * @param eta Learning rate. For the ADAM optimizer, usually suffices to be about 0.01 without much tuning
     * @param beta1 Momentum decay rate, typically about 0.9.
     * @param beta2 Scaling decay rate, typically about 0.999.
     * @param epsilon Small smoothing term, typically about 1.0E-7.
     */
    public ADAM(double eta, double beta1, double beta2, double epsilon) {
        this.eta = eta;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    /**
     * ADAM gradient descent algorithm. The HashMap instances "mMemory" and "sMemory" store the old values of the
     * momentum decay and scaling decay matrices, respectively. Since we always feed the output of this method back
     * into itself in the next epoch, we can set the key to access these old values of m and s as the output of the
     * method itself. If the key does not exist (i.e., running through the first epoch), we create the new key-value
     * mapping. Once we no longer need the references in m and s, we pop the old key-value mapping out of memory and
     * replace it with the new. This algorithm enables us to employ this descent method without losing abstraction.
     *
     * @param toUpdate    The matrix to update
     * @param derivatives The gradients of the matrix for each batch
     * @return The updated matrix.
     */
    public Matrix optimize(Matrix toUpdate, Matrix[] derivatives) {
        Matrix sum = new Matrix(toUpdate.rows(), toUpdate.columns());
        for (Matrix derivative : derivatives) {
            sum = Matrix.sum(sum, derivative);
        }

        Matrix m;
        Matrix s;
        final double sScalar = (1 - beta2) / (derivatives.length * derivatives.length);
        final double mScalar = (beta1 - 1) / derivatives.length;

        if (mMemory.containsKey(toUpdate) && sMemory.containsKey(toUpdate)) {
            m = Matrix.sum(Matrix.scale(beta1, mMemory.get(toUpdate)), Matrix.scale(mScalar, sum));
            s = Matrix.sum(Matrix.scale(beta2, sMemory.get(toUpdate)),
                    Matrix.scale(sScalar, Matrix.hMultiply(sum, sum)));
            mMemory.remove(toUpdate);
            sMemory.remove(toUpdate);
        } else {
            m = Matrix.scale(mScalar, sum);
            s = Matrix.scale(sScalar, Matrix.hMultiply(sum, sum));
        }

        Matrix mHat = Matrix.scale(1 / (1 - Math.pow(beta1, ++i)), m);
        Matrix sHat = Matrix.scale(1 / (1 - Math.pow(beta2, i)), s);

        sHat = sHat.forEach(i -> Math.sqrt(i + epsilon));

        Matrix step = Matrix.sum(toUpdate, Matrix.scale(eta, Matrix.hDivide(mHat, sHat)));
        mMemory.put(step, m);
        sMemory.put(step, s);
        return step;
    }

    public void reset() {
        sMemory.clear();
        mMemory.clear();
    }
}

class ADAGrad extends Optimizer {
    private final double eta;
    private final double epsilon;
    private final HashMap<Matrix, Matrix> memory;

    public ADAGrad(double eta, double epsilon) {
        this.eta = eta;
        this.epsilon = epsilon;
        memory = new HashMap<>();
    }

    /**
     * Momentum gradient descent algorithm. The HashMap instance "memory" stores the old values of the momentum matrix
     * for a given matrix. Since we always feed the output of this method back into itself in the next epoch, we can
     * set the key to access this old value of m as the output of the method itself. If the key does not exist
     * (i.e., running through the first epoch), we create the new key-value mapping. Once we no longer need the old
     * value of m, we dump the old key-value mapping from memory and replace it with the new. This algorithm enables
     * us to employ this descent method without losing abstraction.
     *
     * @param toUpdate    The matrix to update
     * @param derivatives The gradient of the matrix
     * @return The updated matrix.
     */
    public Matrix optimize(Matrix toUpdate, Matrix[] derivatives) {
        Matrix s;
        Matrix sum = new Matrix(toUpdate.rows(), toUpdate.columns());
        for (Matrix derivative : derivatives) { // Summing all the derivatives
            sum = Matrix.sum(sum, derivative);
        }
        if (!memory.containsKey(toUpdate)) {
            s = Matrix.hMultiply(sum, sum);
        } else {
            s = memory.get(toUpdate);
            s = Matrix.sum(s, Matrix.hMultiply(sum, sum));
            memory.remove(toUpdate);
        }

        Matrix sHat = s.forEach(i -> Math.sqrt(i + epsilon));
        Matrix step = Matrix.sum(toUpdate, Matrix.scale(-eta / derivatives.length, Matrix.hDivide(sum, sHat)));
        memory.put(step, s);
        return step;
    }

    public void reset() {
        memory.clear();
    }
}







