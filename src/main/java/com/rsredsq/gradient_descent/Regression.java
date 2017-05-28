package com.rsredsq.gradient_descent;

import java.io.Serializable;

public abstract class Regression implements Serializable {
    public abstract double computeError(Point2D point);
    public abstract double[] computeDerivatives(Point2D point);
    public abstract double[] sumDerivatives(double[] arr1, double[] arr2);
    public abstract double[] getParameters();
    public abstract void setParameters(double[] parameters);

    public double sumErrors(double error1, double error2) {
        return error1 + error2;
    }

}
