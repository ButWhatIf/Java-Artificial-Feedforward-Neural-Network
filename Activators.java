import java.lang.Math;
import java.util.function.Function;

abstract class Activators {
    abstract Matrix activate(Matrix z);

    abstract Matrix prime(Matrix z);
}

class Sigmoid extends Activators {
    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula = i -> 1 / (1 + Math.exp(-i));
        return z.forEach(elementFormula);
    }

    public Matrix prime(Matrix z) { // check this for mathematical accuracy
        Function<Double, Double> elementFormula = i -> 1 / (1 + Math.exp(-i)) * (1 - 1 / (1 + Math.exp(-i)));
        return z.forEach(elementFormula);
    }

}

class ReLU extends Activators {

    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula = i -> Math.max(0.0, i);
        return z.forEach(elementFormula);
    }

    public Matrix prime(Matrix z) {
        Function<Double, Double> elementFormula = i -> i > 0.0 ? 0.0 : 1.0;
        return z.forEach(elementFormula);
    }
}


class tanh extends Activators {
    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula =
                i -> (Math.exp(i) - Math.exp((-1) * i)) / (Math.exp(i) + Math.exp((-1) * i));
        return z.forEach(elementFormula);
    }

    public Matrix prime(Matrix z) {
        Function<Double, Double> elementFormula = i -> Math.pow(1.0 / (Math.exp(i) + Math.exp((-1) * i)), 2);
        return z.forEach(elementFormula);
    }
}


class BinaryStep extends Activators {
    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula = i -> i > 0.0 ? 1.0 : 0.0;
        return z.forEach(elementFormula);
    }

    public Matrix prime(Matrix z) {
        return new Matrix(z.rows(), z.columns()); // Since this constructor creates the zero matrix by default.
    }
}

class LeakyReLU extends Activators {
    private final double slope;

    public LeakyReLU(double slope) {
        this.slope = slope;
    }

    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula = i -> i > 0.0 ? i : slope * i;
        return z.forEach(elementFormula);
    }

    public Matrix prime(Matrix z) {
        Function<Double, Double> elementFormula = i -> i > 0.0 ? 1 : slope;
        return z.forEach(elementFormula);
    }
}

class Softmax extends Activators {

    private double sum(Matrix z) {
        double sum = 0.0;
        for (int i = 0; i < z.rows(); i++) {
            for (int j = 0; j < z.columns(); j++) {
                sum += Math.exp(z.get(i, j));
            }
        }
        return sum;
    }

    public Matrix activate(Matrix z) {
        double sum = sum(z);
        Matrix x = new Matrix(z.rows(), z.columns());
        for (int i = 0; i < z.rows(); i++) {
            for (int j = 0; j < z.columns(); j++) {
                x.setElement(i, j, Math.exp(z.get(i, j)) / sum);
            }
        }
        return x;
    }

    public Matrix prime(Matrix z) {
        double sum = sum(z);;
        return z.forEach(i -> Math.exp(i) / sum * (1 - Math.exp(i) / sum));
    }
}

class Softplus extends Activators {

    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula = i -> Math.log(Math.exp(i) + 1);
        return z.forEach(elementFormula);
    }

    public Matrix prime(Matrix z) {
        Function<Double, Double> elementFormula = i -> Math.exp(i) / (1 + Math.exp(i));
        return z.forEach(elementFormula);
    }
}

class Linear extends Activators {

    public Matrix activate(Matrix z) {
        return z;
    }

    public Matrix prime(Matrix z) {
        return z.forEach(i -> 1.0);
    }
}








