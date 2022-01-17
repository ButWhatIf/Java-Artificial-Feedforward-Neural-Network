import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class Sequential {
    public static final int INPUT_INDEX = 0;
    public static final int OUTPUT_INDEX = 1;
    public static final int WEIGHT_INDEX = 0;
    public static final int BIAS_INDEX = 1;

    private Activators[] activations;
    private final Matrix[] weights;
    private final Matrix[] biases;
    private final ArrayList<Matrix> trainingInputs = new ArrayList<>();
    private final ArrayList<Matrix> trainingOutputs = new ArrayList<>();
    private final int layers;
    private Cost loss;
    private double error;

    /**
     * All parameters are integers representing the node counts for each layer. The total number of ints
     * passed into the vararg parameter list determines the number of layers. From these arguments, two arrays
     * of matrices are generated at random representing the weights and biases for each non-input layer.
     *
     * @param nodeCounts Each int value represents the node count of each layer, respective to the order
     *                   in which they are put
     */
    public Sequential(int... nodeCounts) {
        layers = nodeCounts.length;
        /*
        The input layer has neither a weight matrix nor a bias vector assigned to it, hence there are
        layers - 1 weights and biases. The for-loop generates each weight matrix with
        random entries.
         */
        weights = new Matrix[layers - 1];
        biases = new Matrix[layers - 1];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = new Matrix(nodeCounts[i + 1], nodeCounts[i]);
            biases[i] = new Matrix(nodeCounts[i + 1], 1);
            weights[i].randomize();
            biases[i].randomize();
        }
    }


    /**
     * The parameter list runs parallel to the varargs in the constructor method.
     *
     * @param activations The activations for each layer
     */
    public void setActivations(Activators... activations) {
        if (activations.length != weights.length) {
            System.out.println("Invalid number of activations. Expected: " + weights.length +
                    ". Received: " + activations.length + ".");
            return;
        }
        this.activations = activations;
    }

    /**
     * Trains the model for the total number of epochs
     *
     * @param epochs Total number of epochs
     * @param optimizer Optimization algorithm
     * @param batchSize 1 for stochastic, training set size for batch, and mini-batch for any number in-between
     * @param loss Loss function to evaluate the performance of an epoch
     */
    public void train(int epochs, Optimizer optimizer, int batchSize, Cost loss) {
        if (epochs < 1) {
            System.out.println("Cannot train for less than 1 epoch.");
            return;
        }
        System.out.println("Initiating training sequence.");
        this.loss = loss;
        for (int i = 0; i < epochs; i++) {
            learn(batchSize, optimizer);
            System.out.println("Epoch " + (i + 1) + " complete. Error: " + error / batchSize + ".");
        }
        System.out.println("Training completed successfully.");
    }

    /**
     * Trains the model to the desired accuracy
     *
     * @param cutoff    Cutoff accuracy.
     * @param optimizer Parameter optimizer
     * @param batchSize Size of the mini-batch for stochastic gradient descent.
     * @param loss      Loss function.
     */
    public void train(double cutoff, Optimizer optimizer, int batchSize, Cost loss) {
        System.out.println("Initiating training sequence.");
        this.loss = loss;
        int i = 1;
        double comparator = cutoff + 1;
        while (comparator > cutoff) {
            learn(batchSize, optimizer);
            comparator = error / batchSize;
            System.out.println("Epoch " + i + " complete. Error: " + comparator + ".");
            i++;
        }
        System.out.println("Training completed successfully.");
    }

    /**
     * Applies a gradient descent algorithm specified by the Optimizer parameter.
     *
     * @param batchSize Batch size
     * @param optimizer Preferred optimization method
     */
    private void learn(int batchSize, Optimizer optimizer) {
        error = 0.0;
        // Generating the batch
        Matrix[][] batch = generateBatch(batchSize);
        // Saving the inputs and outputs into the new arrays
        Matrix[] inputBatch = batch[INPUT_INDEX];
        Matrix[] outputBatch = batch[OUTPUT_INDEX];
        // 2D arrays, where each inner array contains the gradients of each example in the batch for a given layer
        Matrix[][] weightBatchGradients = new Matrix[weights.length][batchSize];
        Matrix[][] biasBatchGradients = new Matrix[weights.length][batchSize];
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < batchSize; j++) {
                Matrix[][] gradients = ffbp(inputBatch[j], outputBatch[j]); // Testing an input and getting gradients
                // Storing the gradients for each input for each layer
                weightBatchGradients[i][j] = gradients[WEIGHT_INDEX][i];
                biasBatchGradients[i][j] = gradients[BIAS_INDEX][i];
            }
        }

        // Executing an iteration of the preferred optimization method for each weight and bias for each layer
        for (int i = 0; i < weights.length; i++) {
            weights[i] = optimizer.optimize(weights[i], weightBatchGradients[i]);
            biases[i] = optimizer.optimize(biases[i], biasBatchGradients[i]);
        }
        optimizer.reset(); // Resets the optimizer for the next epoch to avoid heap pollution
    }

    /**
     * During the forward pass, all layer inputs and outputs are stored into arrays. Then an error term is computed,
     * which is used to compute the gradients of the weights and biases according to the backpropagation
     * algorithm of Rumelhart et al.
     *
     * Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986).
     * Learning Representations by Back-Propagating Errors. Nature, 323(6088), 533-536.
     *
     * @param input  Test input vector
     * @param output Test output vector
     * @return Gradients of the weights and biases for each non-input layer, chronologically
     */
    private Matrix[][] ffbp(Matrix input, Matrix output) {
        if (input.rows() != weights[0].columns()) {
            System.out.println("Input vector contains " + input.rows() + " while first layer has "
                    + weights[0].columns() + " nodes.");
            return null;
        } else if (output.rows() != weights[weights.length - 1].rows()) {
            System.out.println("Output vector contains " + output.rows() + " node(s) while last layer has "
                    + weights[weights.length - 1].rows() + " node(s).");
            return null;
        }

        // Keeping track of layer inputs and outputs for backpropagation
        Matrix[] layerOutputs = new Matrix[layers];
        Matrix[] layerInputs = new Matrix[layers];
        layerInputs[0] = input; // The input passes into the input layer
        layerOutputs[0] = input; // The input layer has no activation, so it passes straight through
        Matrix x;

        // Feeding forward
        for (int i = 1; i < layers; i++) {
            x = Matrix.sum(Matrix.multiply(weights[i - 1], layerOutputs[i - 1]), biases[i - 1]);
            layerInputs[i] = x;
            layerOutputs[i] = activations[i - 1].activate(x);
        }

        error += loss.loss(layerOutputs[layerOutputs.length - 1], output); //Finding error here.

        // Backpropagating
        Matrix[] differentialErrors = new Matrix[layers - 1];
        differentialErrors[differentialErrors.length - 1] =
                Matrix.hMultiply(
                        loss.prime(layerOutputs[layerOutputs.length - 1], output),
                        activations[activations.length - 1].prime(layerInputs[layerInputs.length - 1])
                );
        // Shift by +1 in the weights and bias matrices (indices are a nightmare).
        for (int i = layers - 3; i >= 0; i--) {
            Matrix product = Matrix.multiply(weights[i + 1].transpose(), differentialErrors[i + 1]);
            Matrix prime = activations[i + 1].prime(layerInputs[i + 1]);
            differentialErrors[i] = Matrix.hMultiply(product, prime);
        }

        Matrix[] weightGradients = new Matrix[layers - 1];
        for (int i = 0; i < layers - 1; i++) {
            weightGradients[i] = Matrix.multiply(differentialErrors[i], layerOutputs[i].transpose());
        }

        return new Matrix[][]{weightGradients, differentialErrors};
    }


    /**
     * Reads a csv file or txt file, so long as each line contains exactly one pair of inputs
     * and outputs, the inputs and outputs separated by a comma. The input is first, output second.
     * If the input is a vector of individual data points, each data point should be separated
     * by a single space. Likewise for the outputs.
     * <p>
     * Acceptable (Inputs,Outputs):
     * 5 10 15,1 2 3
     * 20 25 30,4 5 6
     * <p>
     * Not Acceptable (Inputs,Outputs):
     * 5, 10 15  (There is a space separating the comma and next number)
     * 5 10 15 (No comma)
     * <p>
     * Each individual input and output is converted into
     * a column vector and stored into an ArrayList for convenience during training.
     *
     * @param path Path to find the data file
     * @throws IOException Thrown if data cannot be read from file.
     */
    public void importData(String path) throws IOException {
        System.out.print("Loading data...");
        FileReader userFile;
        try {
            userFile = new FileReader(path);
        } catch (FileNotFoundException e) {
            System.out.println("File path \"" + path + "\" not recognized.");
            return;
        }

        BufferedReader dataReader = new BufferedReader(userFile);
        String str;
        ArrayList<String> inputVectors = new ArrayList<>();
        ArrayList<String> outputVectors = new ArrayList<>();
        try {
            while ((str = dataReader.readLine()) != null) {
                String[] IOVectors = str.split(",");
                inputVectors.add(IOVectors[0]);
                outputVectors.add(IOVectors[1]);
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            System.out.println("Not every input has an output.");
            return;
        }

        for (String inputs : inputVectors) {
            trainingInputs.add(toVector(inputs));
        }
        for (String outputs : outputVectors) {
            trainingOutputs.add(toVector(outputs));
        }

        String megaBackspace = "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
        System.out.println(megaBackspace + "Data loaded!");
    }

    /**
     * Returns a vectorized form of a String input for reading data from files.
     *
     * @param str Each feature of the vector should be separated by a single space. For example,
     *            the string "1 2 3" is acceptable but not "1,2,3".
     * @return A column vector representation of the parameter. For example, "1 2 3" corresponds to the column vector
     * [1 2 3] (transposed).
     */
    public Matrix toVector(String str) {
        String[] stringVals = str.split(" ");
        Matrix asMatrix = new Matrix(stringVals.length, 1);
        for (int i = 0; i < stringVals.length; i++) {
            try {
                asMatrix.setElement(i, 0, Double.parseDouble(stringVals[i]));
            } catch (NumberFormatException e) {
                System.out.println("Unable to parse double in element " + i + ".");
                return null;
            }
        }

        return asMatrix;
    }

    /**
     * Shuffles the training set and generates a training batch. For internal use in this class.
     *
     * @param batchSize Size of the batch to be generated
     * @return A random batch of inputs and outputs. The index of the batch of inputs is 0, and 1 for the outputs.
     */
    private Matrix[][] generateBatch(int batchSize) {
        this.shuffle(trainingInputs, trainingOutputs);
        Matrix[][] inputOutput = new Matrix[2][batchSize];
        for (int i = 0; i < batchSize; i++) {
            inputOutput[INPUT_INDEX][i] = trainingInputs.get(i);
            inputOutput[OUTPUT_INDEX][i] = trainingOutputs.get(i);
        }
        return inputOutput;
    }

    /**
     * Shuffles two arrays of equal size the same way. We assume they are equal size because we are feeding
     * the input and output data into this shuffle method, and this data has already been prepared by other
     * methods. Essentially a duplication of the Collections.shuffle() method but applied to two arrays
     * for computational efficiency. We implicitly assume both arrays are the same size (which is reasonable
     * given that we are shuffling a training data set). For internal usage in this class.
     *
     * @param list1 Input array
     * @param list2 Output array
     */
    private void shuffle(ArrayList<Matrix> list1, ArrayList<Matrix> list2) {
        Random rand = new Random();
        for (int i = list1.size(); i > 1; i--) {
            int j = rand.nextInt(i);
            Collections.swap(list1, i - 1, j);
            Collections.swap(list2, i - 1, j);
        }
    }

    /**
     * Predicts the output of a given vector.
     *
     * @param input Matrix form of the input vector
     * @return What the model thinks the output should be
     */
    public Matrix predict(Matrix input) {
        Matrix[] layerOutputs = new Matrix[layers];
        layerOutputs[0] = input;
        Matrix x;
        for (int i = 1; i < layers; i++) {
            x = Matrix.sum(Matrix.multiply(weights[i - 1], layerOutputs[i - 1]), biases[i - 1]);
            layerOutputs[i] = activations[i - 1].activate(x);
        }
        return layerOutputs[layers - 1];
    }

    /**
     * Predicts the output of a given vector.
     *
     * @param input Each feature of the vector should be separated by a single space. For example,
     *              the vector "1 2 3" is acceptable but not "1,2,3".
     * @return What the model thinks the output should be
     */
    public Matrix predict(String input) {
        Matrix vector = toVector(input);
        Matrix[] layerOutputs = new Matrix[layers];
        layerOutputs[0] = vector;
        Matrix x;
        for (int i = 1; i < layers; i++) {
            x = Matrix.sum(Matrix.multiply(weights[i - 1], layerOutputs[i - 1]), biases[i - 1]);
            layerOutputs[i] = activations[i - 1].activate(x);
        }
        return layerOutputs[layers - 1];
    }

}

