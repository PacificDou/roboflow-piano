# Play piano using [Roboflow Inference](https://github.com/roboflow/inference)

This project shows how to do object detection using [Roboflow](https://roboflow.com/), and gaze detection model using [Roboflow Inference](https://github.com/roboflow/inference) which is an open source project.
1. We will use an object detection model which was trained for detecting soft drinks on a product shelf. More details about the dataset can be found at: https://universe.roboflow.com/product-recognition-h6t0g/drink-detection.
2. We will run gaze detection provided by [Roboflow Inference](https://github.com/roboflow/inference).
3. The gazed drink will be highlighted, and it's nutrition info will be displayed. These nutrition info are manually collected by the author via Google search.
4. Imagine that each drink is a key of the piano, let's play a sound when you gaze at it! Sound files are from here: https://github.com/py2ai/Piano .
5. Have fun! :)


## How to run it?
1. Start [Roboflow Inference](https://github.com/roboflow/inference) docker container  
   ```
   docker run -p 9001:9001 -d roboflow/roboflow-inference-server-cpu
   ```
2. Download this project to your local and run the following commands
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Replace the `API_KEY` in `src/main.py` with your api_key from Roboflow, here is the guide for generating a key: https://docs.roboflow.com/api-reference/authentication  
   Simply speaking, sing up, create a workspace, goto workspace setting, generate a key.
4. Start the program by:  
   ```
   cd src && python main.py
   ```


## Here is an example

https://github.com/PacificDou/roboflow-piano/assets/13517490/06e193ed-d0ca-4882-85aa-05583661501f



## The horizontal direction seems flipped?
It's really depend on how do you define the relative movement between your head and camera.  
The gaze detection model estimates two angles: yaw and pitch (radian) with the positive direction Right/Up.
That means if you turn your head right/up, the detected yaw/pitch will be positive.
You'll be shown as turning left in the camera view when you actually turn right. 
This is because the laptop camera is like a [mirror](https://www.wtamu.edu/~cbaird/sq/2013/01/05/why-do-mirrors-flip-left-to-right-and-not-up-to-down/).
Feel free to flip the direction by adding/removing the `-` sign of variables `dx, dy` in `main.py`.


## How is the gazing point be calculated?
With the gaze detection model, we can get yaw and pitch (radian).
Here I assume the distance between you and the object is around 1 meter, and you're looking at the camera center when both yaw and pitch are zero.
Then we can easily calculate the horizontal/vertical axis shift in physical metric (see next section).
To convert these values to pixels, I assume all the detected cans have the same width of 66.2mm.


## Formulas for calculating horizontal/vertical shifts
1. Horizontal shift: `DISTANCE_TO_OBJECT * tan(Yaw)`  
2. Vertical shift: `DISTANCE_TO_OBJECT * arccos(Yaw) * tan(Pitch)`  

