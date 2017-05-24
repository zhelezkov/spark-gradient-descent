package com.rsredsq.gradient_descent;

import org.apache.spark.api.java.JavaRDD;
import java.io.Serializable;

public abstract class Regression<T> implements Serializable {

    protected int iterationsCount = 1000;
    protected JavaRDD<T> pointsData;
    protected GradientDescent gradientDescent;

    public Regression(JavaRDD<T> pointsData) {
        this.pointsData = pointsData;
        this.gradientDescent = new GradientDescent(this);
    }

    public abstract double computeError();
    public abstract double[] computeDerivatives();
    public abstract double[] getParameters();
    public abstract void setParameters(double[] parameters);
    public abstract void run();

    public double getLearningRate() {
        return gradientDescent.getLearningRate();
    }

    public Regression setLearningRate(double learningRate) {
        gradientDescent.setLearningRate(learningRate);
        return this;
    }

    public int getIterationsCount() {
        return iterationsCount;
    }

    public Regression setIterationsCount(int iterationsCount) {
        this.iterationsCount = iterationsCount;
        return this;
    }
}
