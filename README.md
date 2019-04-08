# Tiny-YOLOv3 using TensorFlow.js

[`Tiny YOLO`](https://pjreddie.com/darknet/yolo/) object detection with the latest version (v1.0.0) of [Tensorflow.js](https://www.tensorflow.org/js).

## Install
```
npm install tfjs-tiny-yolov3
```

## Usage

### Import module

```javascript
import TinyYoloV3 from 'tfjs-tiny-yolov3';
```

### Initialize and load model

```javascript

const model = new TinyYoloV3();

//Optional settings
const model = new TinyYoloV3({
	nObject = 20, 
	scoreTh = .2,  
	iouTh = .3
});

// Use default models
await model.load()


// or specify path 
await model.load("https://.../model.json")

```

### Run model


#### Get Output Feature Only
```
@param image
	Supported input html element:
	- img
	- canvas
	- video
@param flipHorizontal = true
	flip the image if input source is webcam
```
```javascript
const features = await model.predict(image, flipHorizontal);

```


#### Detect Object and Box

```javascript
const boxes = await model.detectAndBox(image, flipHorizontal);

```

### Output box format

```javascript
{
  top,    // Float
  left,   // Float
  bottom, // Float
  right,  // Float
  height, // Float
  width,  // Float
  score,  // Float
  class   // String, e.g. person
}
```

## Credits
- https://github.com/shaqian/tfjs-yolo
- https://github.com/qqwweee/keras-yolo3
- https://github.com/zqingr/tfjs-yolov3
- https://github.com/ModelDepot/tfjs-yolo-tiny
- https://github.com/allanzelener/YAD2K
