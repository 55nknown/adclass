import sys

from matcher import FBTMatcher

def camera(m):
    while True:
        m.load_camera_frame()
        res = m.match()
        m.visualize()
        if res > 0:
            print(m.samples[res].path)
        else:
            print("Empty")

def sample(m, path):
    m.load_input(path)
    res = m.match()
    if res > 0:
        print(m.samples[res].path)
        m.visualize(wait=True)
    else:
        print("Empty")

def main(args):
    if args.__len__() < 2:
        print("Feature-based Template Matching exmaple with OpenCV written in Python\n\nUsage: python3 example [path]")
        exit(1)

    m = FBTMatcher()
    m.load_samples()
    
    if args[1] == "camera":
        camera(m)
    else:
        sample(m, args[1])

if __name__ == "__main__":
    main(sys.argv)