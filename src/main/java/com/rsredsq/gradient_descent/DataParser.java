package com.rsredsq.gradient_descent;

import java.io.Serializable;

public interface DataParser extends Serializable {
    Point2D parsePointsLine(String line);
}
