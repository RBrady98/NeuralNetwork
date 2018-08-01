/**
 * ImageMatrix
 * Helper class for containing mnist image data
 */
public class ImageMatrix {
    private int[][] pixelValues; //Array for holding pixel values 
    private int label; // What the image is meant to show i.e  3

    public ImageMatrix(int rows, int columns) {
        pixelValues = new int[rows][columns];
    }

    public int getPixelValue(int r, int c) {
        return pixelValues[r][c];
    }

    public int getLabel() {
        return label;
    }

    public void setPixelValues(int r, int c, int val) {
        pixelValues[r][c] = val;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public int[][] getPixelArray() {
        return pixelValues;
    }
}