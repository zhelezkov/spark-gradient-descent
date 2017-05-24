import com.rsredsq.gradient_descent.Point2D;
import org.junit.Test;
import static org.junit.Assert.*;

public class Point2DTest {
    @Test
    public void creationTest() {
        Point2D point = new Point2D();
        assertEquals(0, point.getX(), 0);
        assertEquals(0, point.getY(), 0);
    }
}


