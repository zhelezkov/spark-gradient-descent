package com.rsredsq.gradient_descent;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.regex.Pattern;

public class CSVDataLoader implements DataLoader {
    private static final Pattern SPLIT_PATTERN = Pattern.compile(",");

    private JavaSparkContext jsc;

    public CSVDataLoader(JavaSparkContext jsc) {
        this.jsc = jsc;
    }

    @Override
    public JavaRDD<Point2D> loadPointsData(String path) {
        JavaRDD<String> lines = jsc.textFile(path).cache();
        JavaRDD<Point2D> points = lines.map((str) -> {
            String[] tokens = SPLIT_PATTERN.split(str);
            double x = Double.parseDouble(tokens[0]);
            double y = Double.parseDouble(tokens[1]);
            return new Point2D(x, y);
        });
        return points;
    }
}
