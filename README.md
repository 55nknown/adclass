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
  - This means there should be a model that works for all images, and is trained generally for image comparising and returns a percentage.

## Technologies

- Backend: Node JS (Next)
  - Contains the image comparising API
	- Returns ad information from the database
- Machine learning for embedded systems: Tensor Flow Lite
  - Finds ad frames in an image taken by the mobile client

