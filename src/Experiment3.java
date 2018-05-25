public class Experiment3 extends Experiment {
    public static void main(String strings[]) {
        InputVector[] inputVectors = Utils.getInputVectors("letter-recognition.data");
        Utils.normalizeVectors(inputVectors);

        System.out.println("Initializing Experiment 3...");
        Experiment experiment = new Experiment3(inputVectors);

        System.out.println("Training...");
        (experiment).train();
        (experiment).validate();
    }

    public Experiment3(InputVector[] inputVectors) {
        super(inputVectors, null);
        neuralNetwork = new NeuralNetwork(16 /*input neuron count*/, 100 /*hidden neuron count*/, 26 /*output neuron count*/);
        neuralNetwork.initControlVariables(0.5 /*learning rate*/, 0.2 /*momentum*/, 400 /*max epoch*/, 95 /*desired accuracy*/);
    }

    public void generateTargetSets(String pattern) {
        for (InputVector inputVector : inputVectors) {
            int t[] = new int[26];
            t[inputVector.classChar - 'A'] = 1;
            inputVector.t = t;
        }
    }
}
