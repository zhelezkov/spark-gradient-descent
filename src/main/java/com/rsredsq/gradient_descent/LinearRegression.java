package com.rsredsq.gradient_descent;

import org.apache.spark.api.java.JavaRDD;
import scala.Tuple2;

public class LinearRegression extends Regression<Point2D> {
    private double currentK = 0.0;
    private double currentB = 0.0;

    public LinearRegression(JavaRDD<Point2D> pointsData) {
        super(pointsData);
    }

    public double computeError() {
        double errorsSum = pointsData.map((point) -> {
            double pointX = point.getX();
            double pointY = point.getY();
            double error = Math.pow(pointY - (currentK * pointX + currentB), 2);
            return error;
        }).reduce((error1, error2) -> {
            return error1 + error2;
        });
        return errorsSum / pointsData.count();
    }

    @Override
    public double[] computeDerivatives() {
        Tuple2<Double, Double> gradientTuple = pointsData.mapToPair((point) -> {
            double pointX = point.getX();
            double pointY = point.getY();
            double gradientK = -2 * pointX * (currentB - currentK * pointX + pointY);
            double gradientB = 2 * (currentB - currentK * pointX + pointY);
            return new Tuple2<Double, Double>(gradientK, gradientB);
        }).reduce((tuple1, tuple2) -> {
            double kSum = tuple1._1() + tuple2._1();
            double bSum = tuple1._2() + tuple2._2();
            return new Tuple2<Double, Double>(kSum, bSum);
        });
        double gradientK = gradientTuple._1() / pointsData.count();
        double gradientB = gradientTuple._2() / pointsData.count();

        double[] derives = {gradientK, gradientB};
        return derives;
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

    public void run() {
        for (int i = 0; i < iterationsCount; i++) {
            gradientDescent.step();
        }
    }

}
