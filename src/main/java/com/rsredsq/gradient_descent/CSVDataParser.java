package com.rsredsq.gradient_descent;

import java.util.regex.Pattern;

public class CSVDataParser implements DataParser {
    private static final Pattern SPLIT_PATTERN = Pattern.compile(",");

    @Override
    public Point2D parsePointsLine(String line) {
        String[] tokens = SPLIT_PATTERN.split(line);
        double x = Double.parseDouble(tokens[0]);
        double y = Double.parseDouble(tokens[1]);
        return new Point2D(x, y);
    }
}
