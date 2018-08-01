import java.io.DataInputStream;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
    MNISTParse is a helper class which reads the images from the mnist data set
 */
public class MNISTParse {

    /**
     * MNIST images are read by reading specific bytes from the input files.
     *  Image file format by int, magic number,num of images, num of rows, num of columns,
     * then each unsigned bytes for each pixel value
     * @param imageFilePath String containing path to mnist data file
     * @param labelFilePath String containing path to mnist label file
     * 
     * @return Array of ImageMatrix Objects
     */
    public static ImageMatrix[] readData(String imageFilePath, String labelFilePath) throws IOException{
        //Read the bytes from the image file
        DataInputStream imageInputStream = null;
        DataInputStream labelInputStream = null;
        try {
            imageInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(imageFilePath)));
            labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
            
        } catch (FileNotFoundException e) {
            System.out.println("file not found yo");
            //TODO: handle exception
        }
        //Image file format by byte, magic number,num of images, num of rows, num of columns, pixels
        imageInputStream.readInt(); //skip magic number
        int numOfImages = imageInputStream.readInt();
        int row = imageInputStream.readInt();
        int col = imageInputStream.readInt();

        labelInputStream.readInt(); //skip magic number
        int numOfLabels = labelInputStream.readInt();

        ImageMatrix[] images = null;

        if(numOfImages == numOfLabels) {
            images = new ImageMatrix[numOfImages];

            for (int i = 0; i < numOfImages; i++) {
                ImageMatrix tempMatrix = new ImageMatrix(row, col);
                tempMatrix.setLabel(labelInputStream.readUnsignedByte());
                //Image bytes are row first then column
                for (int r = 0; r < row; r++) {
                    for (int c = 0; c < col; c++) {
                        int color = imageInputStream.readUnsignedByte();
                        tempMatrix.setPixelValues(c, r, color);
                    }
                }

                images[i] = tempMatrix;
            }
        }

        return images;
    }
}