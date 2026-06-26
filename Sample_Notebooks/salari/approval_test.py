from matplotlib.testing.compare import compare_images

if __name__ == "__main__":
    compare_images("approved.pdf", "test.pdf", tol=1)