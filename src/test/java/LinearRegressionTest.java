import com.rsredsq.gradient_descent.LinearRegression;
import com.rsredsq.gradient_descent.Point2D;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class LinearRegressionTest {

    LinearRegression regression;

    @Before
    public void init() {
        regression = new LinearRegression();
        regression.setParameters(new double[] { 1, 0 });
    }

    @Test
    public void computeErrorTest() {
        double error = regression.computeError(new Point2D());
        assertEquals(error, 0, 0.1);
    }

    @Test
    public void computeDerivativesTest() {
        double[] actualDerives = regression.computeDerivatives(new Point2D());
        double[] expectedDerives = new double[] { 0, 0 };
        assertArrayEquals(expectedDerives, actualDerives, 0.1);
    }

    @Test
    public void sumDerivativesTest() {
        double[] actualSum = regression.sumDerivatives(new double[] { 0, 0 }, new double[] { 1, 1 });
        double[] expectedSum = new double[] { 1, 1 };
        assertArrayEquals(expectedSum, actualSum, 0.1);
    }
}
