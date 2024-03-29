import java.util.Arrays;
import java.util.Collections;

public abstract class Experiment {
    protected InputVector[] inputVectors;
    protected InputVector[] Dt;
    protected InputVector[] Dg;
    protected InputVector[] Dv;
    protected NeuralNetwork neuralNetwork;

    protected Experiment(InputVector[] inputVectors, String targetPattern) {
        this.inputVectors = inputVectors;
        randomizeSet(inputVectors);
        generateTargetSets(targetPattern);
        splitTrainingSet();
    }

    protected void splitTrainingSet() {
        int firstSplit = (int)(0.6 * inputVectors.length),
        secondSplit = (int)(0.8 * inputVectors.length);

        Dt = new InputVector[firstSplit];
        Dg = new InputVector[secondSplit - firstSplit];
        Dv = new InputVector[inputVectors.length - secondSplit];

        System.arraycopy(inputVectors, 0, Dt, 0, firstSplit);
        System.arraycopy(inputVectors, firstSplit, Dg, 0, secondSplit - firstSplit);
        System.arraycopy(inputVectors, secondSplit, Dv, 0, inputVectors.length - secondSplit);
    }

    protected void randomizeSet(InputVector[] set) {
        Collections.shuffle(Arrays.asList(set));
    }

    public double train() {
        System.out.println("Training set accuracy: " + neuralNetwork.train(Dt, "Dt"));
        double accuracy =  neuralNetwork.train(Dg, "Dg");
        System.out.println("Generalization set accuracy: " + accuracy);
        return accuracy;
    }

    public double validate() {
        double accuracy = neuralNetwork.validate(Dv);
        System.out.println("Validation set accuracy: " + accuracy);
        return accuracy;
    }
    abstract void generateTargetSets(String pattern);
}
