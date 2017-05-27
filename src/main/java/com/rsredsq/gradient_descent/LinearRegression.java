package com.rsredsq.gradient_descent;

public class LinearRegression implements Regression {
    private double currentK = 0.0;
    private double currentB = 0.0;

    @Override
    public double computeError(Point2D point) {
        double pointX = point.getX();
        double pointY = point.getY();
        double error = Math.pow(pointY - (currentK * pointX + currentB), 2);
        return error;
    }

    @Override
    public double sumErrors(double error1, double error2) {
        return error1 + error2;
    }

    @Override
    public double[] computeDerivative(Point2D point) {
        double pointX = point.getX();
        double pointY = point.getY();
        double deriveK = -2 * pointX * (currentB - currentK * pointX + pointY);
        double deriveB = 2 * (currentB - currentK * pointX + pointY);
        return new double[] { deriveK, deriveB };
    }

    @Override
    public double[] sumDerivatives(double[] arr1, double[] arr2) {
        double kSum = arr1[0] + arr2[0];
        double bSum = arr1[1] + arr2[1];
        return new double[] { kSum, bSum };
    }

    @Override
    public double[] getParameters() {
        double[] params = {currentK, currentB};
        return params;
    }

    @Override
    public void setParameters(double[] parameters) {
        currentK = parameters[0];
        currentB = parameters[1];
    }
}
