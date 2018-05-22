public class Experiment3 extends Experiment {

    public Experiment3(InputVector[] inputVectors) {
        super(inputVectors);
        neuralNetwork = new NeuralNetwork(16, 100, 26);
        neuralNetwork.initControlVariables(0.2, 0.02, 10000, 90);
        neuralNetwork.setInputVectors(inputVectors);
    }

    public double start() {
        return neuralNetwork.train();
    }

    public void generateTargetSets() {
        for (InputVector inputVector : inputVectors) {
            int t[] = new int[26];
            for (int a = 0; a < 26; a++)
                t[a] = (a == inputVector.classChar - 'A'? 0 : 1);
//            t[inputVector.classChar - 'A'] = 1;
            inputVector.t = t;
        }
    }
}
