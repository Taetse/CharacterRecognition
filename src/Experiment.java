import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

public abstract class Experiment {
    protected InputVector[] inputVectors;
    protected InputVector[] Dt;
    protected InputVector[] Dg;
    protected InputVector[] Dv;
    protected NeuralNetwork neuralNetwork;
    public Experiment(InputVector[] inputVectors) {
        this.inputVectors = inputVectors;
        generateTargetSets();
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

    abstract double start();
    abstract void generateTargetSets();
}
