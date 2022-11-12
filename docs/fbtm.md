# Feature-based Template Matching

So my previous solution might have been a _little_ over-engineered. I received the hint, to solve with problem with a technique called [Template Matching](https://en.wikipedia.org/wiki/Template_matching). While template matching is great and all, in this specific use case, it can't really handle all the distortions and transformations in it's basic form. That is why I set out to find a solutions that uses template matching, but could handle harder scenarios that occur in the real life.
Eventually, I came across a method called Feature-based Template Matching, which essentially finds distinct points in images, and compares them.

_Here is an example of Feature-based template matching:_
![example](https://opencv24-python-tutorials.readthedocs.io/en/latest/_images/matcher_result1.jpg)

I created the following [example program](/example/README.md) with the help of the [OpenCV documentation](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html).

## Testing

- According to my tests with a few samples, the example program found the correct ads all the time
- I haven't tested this in real-life scenarios, so it could very well be, that it performs worse, but I've tried to make it more realistic by adding noise, distortions and dirt over the sample images.

## Optimization

- Running the example program, it executes pretty fast already _(less than 400ms on my machine)_, but what could speed it up even more is to **pre-compute the keypoints** on the static ad images and load them back on each run.
- For finding features more easily, a sharpening filter could be applied on the input image. According to my tests, this turned out to be true.
- Another way to improve confidency is to gather more samples and average the results. The only caveat with this method would be the increased bandwidth usage of uploading multiple images, but this could be easily solved by moving the entire computation to [client-side](https://opencv.org/android/).
