import java.lang.Math;
import java.util.function.Function;

abstract class Activators {
    abstract Matrix activate(Matrix z);

    abstract Matrix prime(Matrix z);
}

class Sigmoid extends Activators {
    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula = i -> 1 / (1 + Math.exp(-i));
        z.forEach(elementFormula);
        return z;
    }

    public Matrix prime(Matrix z) { // check this for mathematical accuracy
        Function<Double, Double> elementFormula = i -> 1 / (1 + Math.exp(-i)) * (1 - 1 / (1 + Math.exp(-i)));
        z.forEach(elementFormula);
        return z;
    }

}

class ReLU extends Activators {

    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula = i -> Math.max(0.0, i);
        z.forEach(elementFormula);
        return z;
    }

    public Matrix prime(Matrix z) {
        Function<Double, Double> elementFormula = i -> i > 0.0 ? 0.0 : 1.0;
        z.forEach(elementFormula);
        return z;
    }
}


class tanh extends Activators {
    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula =
                i -> (Math.exp(i) - Math.exp((-1) * i)) / (Math.exp(i) + Math.exp((-1) * i));
        z.forEach(elementFormula);
        return z;
    }

    public Matrix prime(Matrix z) {
        Function<Double, Double> elementFormula = i -> Math.pow(1.0 / (Math.exp(i) + Math.exp((-1) * i)), 2);
        z.forEach(elementFormula);
        return z;
    }
}


class BinaryStep extends Activators {
    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula = i -> i > 0.0 ? 1.0 : 0.0;
        z.forEach(elementFormula);
        return z;
    }

    public Matrix prime(Matrix z) {
        Function<Double, Double> elementFormula = i -> 0.0;
        z.forEach(elementFormula);
        return z;
    }
}

class LeakyReLU extends Activators {
    private double slope1 = 0;
    private double slope2 = 0;

    public LeakyReLU(double slope1, double slope2) {
        this.slope1 = slope1;
        this.slope2 = slope2;
    }

    public Matrix activate(Matrix z) {
        Function<Double, Double> elementFormula = i -> i > 0.0 ? slope2 * i : slope1 * i;
        z.forEach(elementFormula);
        return z;
    }

    public Matrix prime(Matrix z) {
        Function<Double, Double> elementFormula = i -> i > 0.0 ? slope2 : slope1;
        z.forEach(elementFormula);
        return z;
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
        for (int i = 0; i < z.rows(); i++) {
            for (int j = 0; j < z.columns(); j++) {
                z.setElement(i, j, Math.exp(z.get(i, j)) / sum);
            }
        }
        return z;
    }

    public Matrix prime(Matrix z) {
        double sum = sum(z);
        z.forEach(i -> Math.exp(i) / sum * (1 - Math.exp(i) / sum));
        return z;
    }
}








