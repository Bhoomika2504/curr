package com.example.currencydetection;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.MediaPlayer;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.media3.common.util.Log;
import androidx.media3.common.util.UnstableApi;

import com.example.currencydetection.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, confidence;
    ImageView imageView;
    Button picture;
    int imageSize = 224;
    MediaPlayer mediaPlayer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    // Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());
            if (model == null) {
                throw new RuntimeException("Model initialization failed.");
            }


            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer =  ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize* imageSize];
            image.getPixels(intValues,0, image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel = 0;
            for(int i = 0; i <imageSize; i++){
                for(int j = 0; j <imageSize; j++){
                    int val = intValues[pixel++]; //RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f/255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f/255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f/255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidences = 0;
            for (int i = 0; i< confidences.length; i++){
                if(confidences[i]> maxConfidences){
                    maxConfidences = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"10 Rupees","20 Rupees","50 Rupees","100 Rupees","200 Rupees","500 Rupees"};

            result.setText(classes[maxPos]);

            String s = "";
            for(int i = 0; i < classes.length; i++){
                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
            }
            confidence.setText(s);

            // Play audio based on detected currency
            Uri audioUri = null;
            switch (maxPos) {
                case 0: // Currency 10
                    audioUri = Uri.parse("android.resource://" + getPackageName() + "/" + R.raw.currency_10);
                    break;
                case 1: // Currency 20
                    audioUri = Uri.parse("android.resource://" + getPackageName() + "/" + R.raw.currency_20);
                    break;
                case 2: // Currency 50
                    audioUri = Uri.parse("android.resource://" + getPackageName() + "/" + R.raw.currency_50);
                    break;
                case 3: // Currency 100
                    audioUri = Uri.parse("android.resource://" + getPackageName() + "/" + R.raw.currency_100);
                    break;
                case 4: // Currency 200
                    audioUri = Uri.parse("android.resource://" + getPackageName() + "/" + R.raw.currency_200);
                    break;
                case 5: // Currency 500
                    audioUri = Uri.parse("android.resource://" + getPackageName() + "/" + R.raw.currency_500);
                    break;

            }

            // Release previous MediaPlayer instance
            if (mediaPlayer != null) {
                mediaPlayer.release();
            }

            // Initialize new MediaPlayer instance with audio URI
            mediaPlayer = MediaPlayer.create(this, audioUri);

            // Start playing audio
            mediaPlayer.start();

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @OptIn(markerClass = UnstableApi.class) @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK) {
            try {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                if (image == null) {
                    Log.e("MainActivity", "Received image data is null.");
                    return;
                }
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            } catch (Exception e) {
                Log.e("MainActivity", "Error processing image: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Release MediaPlayer resources
        if (mediaPlayer != null) {
            mediaPlayer.release();
            mediaPlayer = null;
        }
    }
}
