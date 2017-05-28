package com.rsredsq.gradient_descent;

public class QuadraticRegression extends Regression {

    private double currentA = 0.0;
    private double currentB = 0.0;
    private double currentC = 0.0;

    @Override
    public double computeError(Point2D point) {
        double pointX = point.getX();
        double pointY = point.getY();
        double error = Math.pow(pointY - (currentA * pointX * pointX + currentB * pointX + currentC), 2);
        return error;
    }

    @Override
    public double[] computeDerivatives(Point2D point) {
        double pointX = point.getX();
        double pointY = point.getY();
        double deriveA = (2 * pointX * pointX) * (pointX * (currentA * pointX + currentB) + currentC);
        double deriveB = 2 * pointX * (pointX * (currentA * pointX + currentB) + currentC);
        double deriveC = 2 * (pointX * (currentA * pointX + currentB) + currentC);
        return new double[] { deriveA, deriveB, deriveC };
    }

    @Override
    public double[] sumDerivatives(double[] arr1, double[] arr2) {
        double aSum = arr1[0] + arr2[0];
        double bSum = arr1[1] + arr2[1];
        double cSum = arr1[2] + arr2[2];
        return new double[] { aSum, bSum, cSum };
    }

    @Override
    public double[] getParameters() {
        return new double[] { currentA, currentB, currentC };
    }

    @Override
    public void setParameters(double[] parameters) {
        currentA = parameters[0];
        currentB = parameters[1];
        currentC = parameters[2];
    }
}
