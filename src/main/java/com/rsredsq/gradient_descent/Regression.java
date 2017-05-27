package com.rsredsq.gradient_descent;

import java.io.Serializable;

public interface Regression extends Serializable {
    double computeError(Point2D point);
    double sumErrors(double error1, double error2);
    double[] computeDerivative(Point2D point);
    double[] sumDerivatives(double[] arr1, double[] arr2);
    double[] getParameters();
    void setParameters(double[] parameters);
}
