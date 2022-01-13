import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Random;

public class DeepLearner {
    private Activators[] activations;
    private Cost loss;
    private Matrix[] weights;
    private Matrix[] biases;
    private ArrayList<Matrix> trainingInputs;
    private ArrayList<Matrix> trainingOutputs;
    private int layers;
    private double error;

    /**
     * Creates a new neural network where the first argument is the cost function, and all subsequent entries
     * are integers representing the node counts for each layer. The total number of ints passed into the
     * vararg parameter list determines the number of layers.
     *
     * @param nodeCounts Each int value represents the node count of each layer, respective to the order
     *                   which they are input
     */
    public DeepLearner(int... nodeCounts) {
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
     * The parameter list runs parallel to the varargs in the Model constructor method.
     *
     * @param activations The activations for each layer of the Model
     */
    public void setActivations(Activators... activations) {
        if (activations.length != weights.length) {
            System.out.println("Invalid number of activations. Expected: " + weights.length +
                    ". Received: " + activations.length + ".");
            return;
        }
        this.activations = activations;
    }

    public void train(int epochs, double learningRate, int batchSize, Cost loss) {
        if (epochs < 1) {
            System.out.println("Cannot train for less than 1 epoch.");
            return;
        }
        System.out.println("Initiating training sequence.");
        this.loss = loss;
        for (int i = 0; i < epochs; i++) {
            learn(batchSize, learningRate);
            System.out.println("Epoch " + (i + 1) + " complete. Error: " + error + ".");
        }
        System.out.println("Training completed successfully.");
    }

    /**
     * Trains the model to the desired accuracy
     *
     * @param cutoff       Cutoff accuracy between 0 and 1. It is recommended to make this value between
     *                     0.8 and 0.9 to avoid over-fitting.
     * @param learningRate Learning rate for the model
     * @param batchSize    Size of the mini-batch for stochastic gradient descent.
     * @param loss         Loss function.
     */
    public void train(double cutoff, double learningRate, int batchSize, Cost loss) {
        System.out.println("Initiating training sequence.");
        this.loss = loss;
        int i = 1;
        double comparator = cutoff + 1;
        while (comparator > cutoff) {
            learn(batchSize, learningRate);
            comparator = error;
            System.out.println("Epoch " + i + " complete. Error: " + comparator + ".");
            i++;
        }
        System.out.println("Training completed successfully.");
    }

    private void learn(int batchSize, double learningRate) {
        Matrix[][] batch = generateBatch(batchSize);
        Matrix[] inputBatch = batch[0];
        Matrix[] outputBatch = batch[1];
        Matrix[][] weightGradients = new Matrix[batchSize][layers - 1];
        Matrix[][] biasGradients = new Matrix[batchSize][layers - 1];

        // Finds the weight gradients for each batch.
        for (int i = 0; i < inputBatch.length; i++) {
            error = 0.0;
            Matrix[][] backprop = ffbp(inputBatch[i], outputBatch[i]);
            weightGradients[i] = backprop[0];
            biasGradients[i] = backprop[1];
        }
        // Sums all the weight gradients for each batch layer by layer
        Matrix[] summedWeightGradients = new Matrix[layers - 1];
        Matrix[] summedBiasGradients = new Matrix[layers - 1];
        for (int j = 0; j < layers - 1; j++) {
            summedWeightGradients[j] = new Matrix(weights[j].rows(), weights[j].columns());
            summedBiasGradients[j] = new Matrix(biases[j].rows(), 1);
            for (int i = 0; i < batchSize; i++) {
                summedWeightGradients[j] = Matrix.sum(summedWeightGradients[j], weightGradients[i][j]);
                summedBiasGradients[j] = Matrix.sum(summedBiasGradients[j], biasGradients[i][j]);
            }
        }
        // Gradient descent
        for (int i = 0; i < layers - 1; i++) {
            Matrix weightLoss = Matrix.scale(-learningRate / batchSize, summedWeightGradients[i]);
            Matrix biasLoss = Matrix.scale(-learningRate / batchSize, summedBiasGradients[i]);
            weights[i] = Matrix.sum(weights[i], weightLoss);
            biases[i] = Matrix.sum(biases[i], biasLoss);
        }
    }

    /**
     * During the forward pass, all layer inputs and outputs are stored into arrays. The data
     * stored in said arrays are
     *
     * @param input  Test input vector
     * @param output Test output vector
     * @return Gradients of the weights and biases.
     */
    public Matrix[][] ffbp(Matrix input, Matrix output) {
        if (input.rows() != weights[0].columns()) {
            System.out.println("Input vector contains " + input.rows() + " while first layer has "
                    + weights[0].columns() + " nodes.");
            return null;
        } else if (output.rows() != weights[weights.length - 1].rows()) {
            System.out.println("Output vector contains " + output.rows() + " while last layer has "
                    + weights[weights.length - 1].rows() + " nodes.");
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
                        loss.lossDerivative(output, layerOutputs[layers - 1]),
                        activations[activations.length - 1].prime(layerInputs[layers - 1])
                );
        // Shift by +1 in the weights and bias matrices (indices are a nightmare).
        for (int i = differentialErrors.length - 2; i >= 0; i--) {
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
     * Acceptable (Inputs,Outputs):
     * 5 10 15,1 2 3
     * 20 25 30,4 5 6
     * Not Acceptable (Inputs,Outputs):
     * 5, 10 15  (There is a space separating the comma and next number)
     * 5 10 15 (No comma)
     *
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
        while ((str = dataReader.readLine()) != null) {
            String[] IOVectors = str.split(",");
            inputVectors.add(IOVectors[0]);
            outputVectors.add(IOVectors[1]);
        }

        if (inputVectors.size() != outputVectors.size()) {
            System.out.println("Not every input has an output.");
            return;
        }

        ArrayList<String[]> inputToParse = new ArrayList<>();
        ArrayList<String[]> outputToParse = new ArrayList<>();
        int inputVectorLength = inputVectors.get(0).split(" ").length;
        int outputVectorLength = outputVectors.get(0).split(" ").length;
        for (String strings : inputVectors) {
            String[] stringedInputs = strings.split(" ");
            if (stringedInputs.length != inputVectorLength) {
                System.out.println("Not all inputs have same row count.");
                return;
            }
            inputToParse.add(stringedInputs);
        }
        for (String strings : outputVectors) {
            String[] stringedOutputs = strings.split(" ");
            if (stringedOutputs.length != outputVectorLength) {
                System.out.println("Not all outputs have same row count.");
                return;
            }
            outputToParse.add(stringedOutputs);
        }

        ArrayList<double[][]> inputElements = new ArrayList<>();
        ArrayList<double[][]> outputElements = new ArrayList<>();
        trainingInputs = new ArrayList<>();
        trainingOutputs = new ArrayList<>();
        try {
            for (String[] inputs : inputToParse) {
                double[][] elementData = new double[inputVectorLength][1];
                int i = 0;
                for (String number : inputs) {
                    double[] myNum = {Double.parseDouble(number)};
                    elementData[i] = myNum;
                    i++;
                }
                inputElements.add(elementData);
            }
            for (String[] outputs : outputToParse) {
                double[][] elementData = new double[outputVectorLength][1];
                int i = 0;
                for (String number : outputs) {
                    double[] myNum = {Double.parseDouble(number)};
                    elementData[i] = myNum;
                    i++;
                }
                outputElements.add(elementData);
            }
        } catch (NumberFormatException e) {
            System.out.println("Ensure that each data point is a numerical value.");
        }

        for (double[][] elements : inputElements) {
            trainingInputs.add(new Matrix(elements));
        }
        for (double[][] elements : outputElements) {
            trainingOutputs.add(new Matrix(elements));
        }

        String megaBackspace = "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b";
        System.out.println(megaBackspace + "Data loaded!");
    }

    /**
     * Saves model to a new directory in the designated path
     *
     * @param pathway Location to make new
     */
    public void saveModel(String pathway) {
        Path path = Paths.get(pathway);
        try {
            Files.createDirectories(path);

            System.out.println("Saved model to " + pathway + ".");
        } catch (IOException e) {
            System.out.println("Unable to save model to the designated path.");
        }

    }

    /**
     * Saves model to project folder
     */
    public void saveModel() {

    }

    private Matrix[][] generateBatch(int batchSize) {
        shuffle(trainingInputs, trainingOutputs);
        Matrix[][] inputOutput = new Matrix[2][batchSize];
        for (int i = 0; i < batchSize; i++) {
            inputOutput[0][i] = trainingInputs.get(i);
            inputOutput[1][i] = trainingInputs.get(i);
        }
        return inputOutput;
    }

    /**
     * Shuffles two arrays of equal size the same way. We assume they are equal size because we are feeding
     * the input and output data into this shuffle method, and this data has already been prepared by other
     * methods.
     *
     * @param list1 Input array
     * @param list2 Output array
     */
    private void shuffle(ArrayList<Matrix> list1, ArrayList<Matrix> list2) {
        Random rand = new Random();
        for (int i = 0; i < list1.size(); i++) {
            int newIndex = rand.nextInt(list1.size());
            Matrix temp1 = list1.get(newIndex);
            list1.set(newIndex, list1.get(i));
            list1.set(i, temp1);
            // Repeating process for list2
            Matrix temp2 = list2.get(newIndex);
            list2.set(newIndex, list2.get(i));
            list2.set(i, temp2);
        }
    }

    public ArrayList<Matrix> getInputs() {
        return trainingInputs;
    }

    public ArrayList<Matrix> getOutputs() {
        return trainingOutputs;
    }

}

