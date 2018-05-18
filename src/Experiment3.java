public class Experiment3 extends Experiment {

    public Experiment3(InputVector[] inputVectors) {
        super(inputVectors);
        neuralNetwork = new NeuralNetwork(16, 1000, 26);
        neuralNetwork.initControlVariables(0.9, 0.1, 100, 90);
        neuralNetwork.setInputVectors(inputVectors);
    }

    public double start() {
        return neuralNetwork.train();
    }

    public void generateTargetSets() {
        for (InputVector inputVector : inputVectors) {
            int t[] = new int[26];
            t[inputVector.classChar - 'A'] = 1;
            inputVector.t = t;
        }
    }
}
