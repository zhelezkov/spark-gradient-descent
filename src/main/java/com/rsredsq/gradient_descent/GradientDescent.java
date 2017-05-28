package com.rsredsq.gradient_descent;

import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import org.apache.spark.broadcast.Broadcast;

public class GradientDescent {
    private static final int DEFAULT_ITERATIONS_COUNT = 1000;
    private static final double DEFAULT_LEARNING_RATE = 0.0001;

    private JavaSparkContext jsc;
    private Regression regression;
    private JavaRDD<Point2D> pointsData;
    private Broadcast<Long> pointsDataCount;

    private int iterationsCount = DEFAULT_ITERATIONS_COUNT;
    private double learningRate = DEFAULT_LEARNING_RATE;

    public GradientDescent(JavaSparkContext jsc, JavaRDD<Point2D> pointsData) {
        this.jsc = jsc;
        this.pointsData = pointsData;
        pointsDataCount = jsc.broadcast(pointsData.count());
    }

    public void run() {
        if (regression == null) throw new RuntimeException("No regression");

        for (int i = 0; i < iterationsCount; i++) {
            double[] parameters = regression.getParameters();

            double[] derivatives = pointsData.map(regression::computeDerivatives).reduce(regression::sumDerivatives);

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

    public int getIterationsCount() {
        return iterationsCount;
    }

    public GradientDescent setIterationsCount(int iterationsCount) {
        this.iterationsCount = iterationsCount;
        return this;
    }

    public double[] getRegressionParameters() {
        return regression.getParameters();
    }

    public Regression getRegression() {
        return regression;
    }

    public GradientDescent setRegression(Regression regression) {
        this.regression = regression;
        return this;
    }

    public static void main(String[] args) {
        if (args.length < 1) throw new IllegalArgumentException("Not enough arguments");

        SparkConf conf = new SparkConf();
        conf.setAppName("Gradient Descent");

        JavaSparkContext jsc = new JavaSparkContext(conf);

        DataParser csvDataParser = new CSVDataParser();
        JavaRDD<Point2D> pointsData = jsc.textFile(args[0]).map(csvDataParser::parsePointsLine).cache();

        GradientDescent gradientDescent = new GradientDescent(jsc, pointsData);

        if (args.length >= 2) {
            int type = Integer.parseInt(args[1]);
            if (type == 1) gradientDescent.setRegression(new LinearRegression());
            if (type == 2) gradientDescent.setRegression(new QuadraticRegression());
        } else {
            gradientDescent.setRegression(new LinearRegression());
        }

        if (args.length >= 3) {
            gradientDescent.setIterationsCount(Integer.parseInt(args[2]));
        }

        if (args.length >= 4) {
            gradientDescent.setLearningRate(Double.parseDouble(args[3]));
        }

        gradientDescent.run();

        Regression regression = gradientDescent.getRegression();

        double[] parameters = regression.getParameters();

        for (double param : parameters) {
            System.out.println(param);
        }

        jsc.stop();
    }
}