public class Experiment3 extends Experiment {

    public Experiment3(InputVector[] inputVectors) {
        super(inputVectors, null);
        neuralNetwork = new NeuralNetwork(16, 100, 26);
        neuralNetwork.initControlVariables(0.3, 0.05, 10000, 90);
    }

    public void generateTargetSets(String pattern) {
        for (InputVector inputVector : inputVectors) {
            int t[] = new int[26];
            t[inputVector.classChar - 'A'] = 1;
            inputVector.t = t;
        }
    }
}
