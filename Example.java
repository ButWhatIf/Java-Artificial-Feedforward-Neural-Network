import java.io.IOException;
import java.io.PrintWriter;

public class Example {

    public static void main(String[] args) throws IOException {
        // Creating a MLP with 1 input, 1 output, and 2 hidden layers with 64 and 32 nodes respectively
        Sequential toy = new Sequential(1, 64, 32, 1);
        // Setting activations for the hidden and output layers
        toy.setActivations(new Sigmoid(), new Sigmoid(), new Linear());
        // Loading the training dataset
        toy.importData("C:\\training_sample.txt");

        /*
        Training the MLP for 50,000 batches using a mini-batch momentum gradient descent with learning rate
        0.001, momentum decay rate of 0.90, and batch size of 10, evaluated with mean average error.
         */
        toy.train(50000, new Momentum(0.001, 0.90), 10, new MeanAverageError());

        // Writing the predicted outputs using this.predict() to a file to compare
        PrintWriter printer = new PrintWriter("C:\\computer_predicted.txt");
        for (int i = -10; i < 10; i++) {
            printer.println((double) i/2 + "," + toy.predict(Double.toString((double) i / 2)));
        }
        printer.close();
    }
}
