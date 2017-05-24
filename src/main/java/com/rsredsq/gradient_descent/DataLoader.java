package com.rsredsq.gradient_descent;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public interface DataLoader {
    JavaRDD<Point2D> loadPointsData(String path);
}
