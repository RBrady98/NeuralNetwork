import java.util.Arrays;
import java.util.Random;
/**
 * NeuralNetork
 */
public class NeuralNetork {

    private double[][] output;
    private double[][][] weights;
    private double[][] bias;

    private double[][] errorSignal; //used in calculating cost function
    private double[][] outputDerivative;

    public final int[] LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;
    
    /**
     * Class Constructor
     * @param layerSizes int array containing amount of neurons in each layer  
     */
    NeuralNetork(int[] layerSizes) {
        LAYER_SIZES = layerSizes;
        INPUT_SIZE = layerSizes[0];
        OUTPUT_SIZE = layerSizes[layerSizes.length - 1];
        NETWORK_SIZE = layerSizes.length;

        this.output = new double[NETWORK_SIZE][];
        this.errorSignal = new double[NETWORK_SIZE][];
        this.outputDerivative = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];

        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[LAYER_SIZES[i]];
            this.errorSignal[i] = new double[LAYER_SIZES[i]];
            this.outputDerivative[i] = new double[LAYER_SIZES[i]];
            this.bias[i] = initializeBiasArray(LAYER_SIZES[i]);

            if(i > 0) {
                this.weights[i] = initializeWeightsArray(LAYER_SIZES[i], LAYER_SIZES[i - 1]);
            }
        }
    }

    /**
     * The calculate method passes the inputs through the network and calculates the output for 
     * each neuron is every layer. The output of each neuron is normalizes
     *  using the sigmoid function
     * 
     * @param input double array containing the input values for the first layer of neurons
     * @return double array containing the outputs of the network output layer 
     */
    public double[] calculate(double[] input) {
        if (input.length != this.INPUT_SIZE) {
            return null;
        }

        this.output[0] = input;
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for (int previousNeuron = 0; previousNeuron < LAYER_SIZES[layer - 1]; previousNeuron++) {
                    sum += output[layer - 1][previousNeuron] * weights[layer][neuron][previousNeuron];
                }
                sum += bias[layer][neuron];
                output[layer][neuron] = sigmoid(sum);
                outputDerivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);
            }
        }

        return output[NETWORK_SIZE - 1];
    }

    /**
     * Implementation of the sigmoid function
     * @param x function input
     * @return output of the function
     */
    private double sigmoid(double x) {
        return 1d / (1 + Math.exp(-x));
    }

    /**
     * Handles the stochastic gradient approach to training the network by calculating the output,
     * calculating the error of each neuron then adding the changes to an array to be later
     * used to update the weights and biases of each neuron
     */
    public double[][] trainByBatch(double[] input, double[] target, double[][] meanWeightChange) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
            return null;
        }
        calculate(input);
        backpropError(target); //calculate the error for each neuron
        for (int i = 0; i < meanWeightChange.length; i++) {
            for (int j = 0; j < meanWeightChange[i].length; j++) {
                //sum all neuron errors, later used after averaging each value
                meanWeightChange[i][j] += errorSignal[i][j];
            }
        }
        return meanWeightChange;
    }

    /**
     * Calculates the error of each node using error signal and output derivative method. 
     *  Sets the error signla of each node in the errorSignal array;
     * 
     * @param targetValues the target vector for the desired output
     */
    public void backpropError(double[] targetValues) {
        for (int neuron = 0; neuron < LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
            errorSignal[NETWORK_SIZE - 1][neuron] = (targetValues[neuron] - output[NETWORK_SIZE - 1][neuron]) 
                    * outputDerivative[NETWORK_SIZE - 1][neuron];
        }

        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < LAYER_SIZES[layer + 1]; nextNeuron++) {
                    sum += weights[layer + 1][nextNeuron][neuron] * errorSignal[layer + 1][nextNeuron];
                }
                errorSignal[layer][neuron] = sum * outputDerivative[layer][neuron];
            }
        }
    }

    /**
     * Updates the weight of every neuron with a new adjusted wieght based onthe error signal.
     * Also adjusts biases
     */
    public void updateWeigths(double eta) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {

                double delta = (eta) * errorSignal[layer][neuron];
                bias[layer][neuron] += delta;

                for (int prevNeuron = 0; prevNeuron < LAYER_SIZES[layer - 1]; prevNeuron++) {
                    weights[layer][neuron][prevNeuron] += delta * output[layer - 1][prevNeuron];
                        
                }
            }
        }
    }

    /**
     * Calculates the mean square error of the network
     * @param output values of the output layer
     * @param target target of the output layer
     * @return Mean Square Error
     */
    public double mse(double[] output, double[] target) {
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            sum += Math.pow(output[i] - target[i], 2);
        }
        sum = sum / 10d;
        return sum;
    }

    /**
     * Sets all values in the bias array to a random values between 0 and 1
     * @param size length of the array to initialise
     * @return bias array
     */
    private double[] initializeBiasArray(int size) {
        double[] ar = new double[size];
        for (int i = 0; i < ar.length; i++) {
            ar[i] = Math.random();
        }

        return ar;
    }

    /**
     * Sets all values in the bias array to a random values between -1 and 1
     * @param sizeX width of the array to initialise
     * @param sizeY height of the array to initialise
     * @return weights array
     */
    private double[][] initializeWeightsArray(int sizeX, int sizeY) {
        double[][] ar = new double[sizeX][sizeY];
        Random rand = new Random();
        for (int x = 0; x < sizeX; x++) {
            for (int y = 0; y < sizeY; y++) {
                ar[x][y] = rand.nextGaussian();
            }
        }

        return ar;
    }

    /**
     * Trains the network using the stochastic gradient descent method using batches of data
     * @param data Array of ImageMatrix objects containing the MNIST image data
     * @param miniBatchsize size of the mini batches
     * @param learningRate learning rate of the network
     * @param epochs NUmber of times to repeat the whole training process
     */
    public void trainNetork(ImageMatrix[] data, int miniBatchSize, double learningRate, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double[][] meanWeightChanges = errorSignal;
            for (int x = 0; x < data.length / miniBatchSize; x++) {
                for (int i = 0; i < miniBatchSize; i++) {
                    ImageMatrix currentMatrix = data[(x * miniBatchSize) + i];
                    double[] input = getInputMatrixAsVector(currentMatrix.getPixelArray());
                    double[] target = getTargetArray(currentMatrix.getLabel());
                    meanWeightChanges = trainByBatch(input, target, meanWeightChanges);
                    // System.out.println(mse(getOutputLayerValues(), target));
                }
                errorSignal = normaliseMeanWeightChanges(meanWeightChanges, miniBatchSize);
                updateWeigths(learningRate);
            }    
        }
    }

    /**
     * Returns the average of the weight changes over a training period
     * @param meanWeightChanges array containing the weight changes over multiple training loops
     * @param miniBatchSize size of the mini batch used during training
     */
    private double[][] normaliseMeanWeightChanges(double[][] meanWeightChanges, int miniBatchSize) {
        for (int i = 0; i < meanWeightChanges.length; i++) {
            for (int j = 0; j < meanWeightChanges[i].length; j++) {
                // System.out.println("Pre normalisation: " + meanWeightChanges[i][j]);
                meanWeightChanges[i][j] = meanWeightChanges[i][j] / (double)miniBatchSize;
                // System.out.println("Post normalisation: " + meanWeightChanges[i][j]);
            }
        }
        return meanWeightChanges;
    }

    /**
     * Returns an array containing the target value in the correct format of an array with the 
     * target numbers index set to 1 and the rest being 0s e.g [0,0,0,1,0,0,0,0]
     * @param targetNum Desired output of the input
     * @return target array in the correct format
     */
    public double[] getTargetArray(int targetNum) {
        double[] ar = new double[10];
        ar[targetNum] = 1;
        return ar;
    }

    /**
     * Converts the pixel values from the MNIST data from a 2d x,y array to a vector for input to 
     * the network
     * @param inputArray array containing pixel values of the MNIST image
     */
    public double[] getInputMatrixAsVector(int[][] inputArray) {
        double[] ar = new double[784];
        int index = 0;
        for (int x = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                ar[index] = inputArray[x][y] / 255d;
                index++;
            }
        }
        return ar;
    }

    /**
     * Returns the networks guess for a certain MNIST image
     * @param data ImageMatrix Object
     * @return Ouputs of the output later neurons
     */
    public double[] testNetwork(ImageMatrix data) {
        double[] input = getInputMatrixAsVector(data.getPixelArray());
        return calculate(input);
    }

    /**
     * Returns what the network thinks the MNIST image is meant to be
     * @param output outputs of the output layer neurons
     * @return integer value from 0-9 for what the network thinks the value is
     */
    public int networkGuess(double[] output) {
        double currentLargest = -1;
        int index = 0;
        for (int i = 0; i < 10; i++) {
            if(Math.max(currentLargest, output[i]) > currentLargest) {
                currentLargest = output[i];
                index = i;
            }
        }
        return index;
    }

    /**
     * Iterates through an array of ImageMatrix objects, passing each into the network and getting the 
     * networks guess as an output. Then checks if the network guess is equal to what the image label is
     * then outputs the correct number of guesses.
     * @param inputData array of ImageMatrixObjects
     */
    public void checkAccuracyOfNetwork(ImageMatrix[] inputData) {
        int correctGuesses = 0;
        for (ImageMatrix matrix : inputData) {
            double[] output = testNetwork(matrix);
            if(matrix.getLabel() == networkGuess(output)) {
                correctGuesses++;
            }
        }
        System.out.println("Correct guesses: " + correctGuesses + "/" + inputData.length);
    }

    /**
     * Main method parses the MNIST data, trains the network and also tests to see how accurate the 
     * network is.
     * @param args command line arguments 
     */
    public static void main(String[] args) {
        ImageMatrix[] trainingData = null;
        ImageMatrix[] testData = null;
        try {
            trainingData = MNISTParse.readData("data\\train-images.idx3-ubyte", "data\\train-labels.idx1-ubyte");
            testData = MNISTParse.readData("data\\t10k-images.idx3-ubyte", "data\\t10k-labels.idx1-ubyte");
            
        } catch (Exception e) {
            e.printStackTrace();
        }

        int[] layers = {784, 25, 10};
        NeuralNetork net = new NeuralNetork(layers);

        net.trainNetork(trainingData, 1, 0.01, 5);

        net.checkAccuracyOfNetwork(testData);
    }

}