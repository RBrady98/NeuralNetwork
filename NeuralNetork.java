import java.util.Arrays;
import java.util.Random;
/**
 * NeuralNetork
 */
public class NeuralNetork {

    private double[][] output;
    private double[][][] weights;
    private double[][] bias;

    private double[][] errorSignal;
    private double[][] outputDerivative;

    public final int[] LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int NETWORK_SIZE;
    
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

    private double sigmoid(double x) {
        return 1d / (1 + Math.exp(-x));
    }

    public void train(double[] input, double[] target, double eta) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) {
            return;
        }
        calculate(input);
        backpropError(target);
        updateWeigths(eta);
    }

    public void backpropError(double[] targetValues) {
        for (int neuron = 0; neuron < LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
            errorSignal[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - targetValues[neuron]) 
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

    public void updateWeigths(double eta) {
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {

                double delta = -(eta) * errorSignal[layer][neuron];
                bias[layer][neuron] += delta;

                for (int prevNeuron = 0; prevNeuron < LAYER_SIZES[layer - 1]; prevNeuron++) {
                    //weights[layer][neuron][prevNeuron]
                    weights[layer][neuron][prevNeuron] += delta * output[layer - 1][prevNeuron];
                }
            }
        }
    }

    private double[] initializeBiasArray(int size) {
        double[] ar = new double[size];
        for (int i = 0; i < ar.length; i++) {
            ar[i] = Math.random();
        }

        return ar;
    }

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
    public static void main(String[] args) {
        int[] layers = {4, 1, 3, 4};
        NeuralNetork net = new NeuralNetork(layers);
        double[] inputs = {0.1, 0.5, 0.2, 0.9};
        double[] target = {0, 1, 0, 0};

        for (int i = 0; i < 1000; i++) {
            net.train(inputs, target, 0.3);
        }
        
        double[] newInputs = {0.3, 0.4, 0.9, 0.2};
        double[] output = net.calculate(newInputs);
        System.out.println(Arrays.toString(output));
    }

}