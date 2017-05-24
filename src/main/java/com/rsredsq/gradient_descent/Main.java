package com.rsredsq.gradient_descent;

import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;

import java.util.concurrent.CountDownLatch;

public class Main {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Simple Application").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        DataLoader csvDataLoader = new CSVDataLoader(jsc);
        JavaRDD<Point2D> pointsData = csvDataLoader.loadPointsData("data.csv");
        LinearRegression regression = new LinearRegression(pointsData);
        regression.setLearningRate(0.0001).setIterationsCount(1000);

        System.out.println(regression.computeError());

        regression.run();

        double[] derives = regression.getParameters();

        System.out.println(derives[0] + " " + derives[1] + " " + regression.computeError());

        try {
            new CountDownLatch(1).await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        jsc.stop();
    }
}