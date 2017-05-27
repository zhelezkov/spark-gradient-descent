package com.rsredsq.gradient_descent;

import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import org.apache.spark.broadcast.Broadcast;

public class GradientDescent {

    private JavaSparkContext jsc;
    private Regression regression;
    private JavaRDD<Point2D> pointsData;
    private Broadcast<Long> pointsDataCount;

    private double learningRate = 0.0001;
    private double iterationsCount = 1000;

    public GradientDescent(JavaSparkContext jsc, JavaRDD<Point2D> pointsData) {
        this.jsc = jsc;
        this.pointsData = pointsData;
        pointsDataCount = jsc.broadcast(pointsData.count());

        regression = new LinearRegression();

        run();
    }

    public void run() {

        for (int i = 0; i < iterationsCount; i++) {
            double[] parameters = regression.getParameters();

            double[] derivatives = pointsData.map(regression::computeDerivative).reduce(regression::sumDerivatives);

            for (int j = 0; j < derivatives.length; j++) {
                derivatives[j] /= pointsDataCount.getValue();
            }

            for (int j = 0; j < parameters.length; j++) {
                parameters[j] = parameters[j] - learningRate * derivatives[j];
            }

            regression.setParameters(parameters);
        }
    }

    public double computeError() {
        return pointsData.map(regression::computeError).reduce(regression::sumErrors) / pointsDataCount.getValue();
    }

    public double getLearningRate() {
        return learningRate;
    }

    public GradientDescent setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public double getIterationsCount() {
        return iterationsCount;
    }

    public GradientDescent setIterationsCount(double iterationsCount) {
        this.iterationsCount = iterationsCount;
        return this;
    }

    public double[] getRegressionParameters() {
        return regression.getParameters();
    }

    //TODO parse arguments
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Gradient Descent");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        DataParser csvDataParser = new CSVDataParser();
        JavaRDD<Point2D> pointsData = jsc.textFile("data1.csv").map(csvDataParser::parsePointsLine).cache();

        GradientDescent gradientDescent = new GradientDescent(jsc, pointsData);

        jsc.stop();
    }
}