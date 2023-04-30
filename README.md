# ImageCompressionPipeline
In this repo, I developed a python script which demonstrates a pipeline for image compression. 

The way this pipline works is demonstrated in this diagram:
![image](https://user-images.githubusercontent.com/82244228/235373106-d1fa8216-f6e2-41bf-b680-d720de011c01.png)

After the image is inputted into the pipeline, it is then downsampled using 2 different techniques. The inputted image is first converted from RGB to the YUV color space. Then we have 2x downsample on the Y channel, and 4x downsample on the U and V channels. This is because most information exists in the Y channel and we retain the most information there:
![image](https://user-images.githubusercontent.com/82244228/235373199-b64a95d7-6172-4c30-860d-6f1afdf8c51e.png)

After the image is downsampled, we can do the reverse operation to upsample the image back at the decoders side:
![image](https://user-images.githubusercontent.com/82244228/235373223-637e3116-3f85-4a4b-b4da-4f8d826754fd.png)

This is a comparison of the input vs the output of this pipeline:
![image](https://user-images.githubusercontent.com/82244228/235373246-1c83a93a-d2a2-45ed-9501-11ffd93b22ae.png)

