# Feature-based Template Matching

So my previous solution might have been a _little_ over-engineered. I received the hint, to solve with problem with a technique called [Template Matching](https://en.wikipedia.org/wiki/Template_matching). While template matching is great and all, in this specific use case, it can't really handle all the distortions and transformations in it's basic form. That is why I set out to find a solutions that uses template matching, but could handle harder scenarios that occur in the real life.
Eventually, I came across a method called Feature-based Template Matching, which essentially finds distinct points in images, and compares them.
