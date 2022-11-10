# AD Classification NN

1. Find the frame of the ad

   - Use some TFLite model

2. Extract ad image

   - Crop image based on results

3. Upload ad image to the server

4. Compare uploaded image with the stored references

5. Find the correct ad based on a threshold

6. Return ad information to the mobile client

## Identification Model

Identifying the ad from the references could work two ways:

1. Train the classification model each time after adding a new ad

   - This is very resource intensive and has many limitations

2. Get some score based value on how much the images match

   - This means there should be a model that works for all images, and is trained generally for image comparison and returns a percentage.

## Technologies used

- Node JS + TensorFlow.js

  - Used on the server side
  - Contains the image comparison API
  - Returns ad information from the database

- Kotlin + Tensor Flow Lite

  - Used on the client side
  - Finds ad frames in an image taken by the mobile client

## Developemnt workflow

1. Collect training data for the NNs
2. Train the frame-finder model on the collected images
3. Train the image comparison model on some reference ads
4. Create an ad identification backend with TF.js
5. Integrate the ad identification client and the TFLite frame-finder

## QA

- A clear guiding should be displayed for the user for taking an optimal pictuer of the ad
- The model should find the correct ad image on the first or second guess
- If the results are not good enough the user should be prompted to retry, or select the correct ad between two or three possible matches
