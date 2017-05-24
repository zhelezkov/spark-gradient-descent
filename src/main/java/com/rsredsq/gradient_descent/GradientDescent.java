package com.rsredsq.gradient_descent;

import java.io.Serializable;

public class GradientDescent implements Serializable {

    private double learningRate = 0.0001;
    private Regression regression;

    public GradientDescent(Regression regression) {
        this.regression = regression;
    }

    public void step() {
        double[] parameters = regression.getParameters();

        double[] derives = regression.computeDerivatives();

        for (int i = 0; i < parameters.length; i++) {
            parameters[i] = parameters[i] - learningRate * derives[i];
        }

        regression.setParameters(parameters);
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
