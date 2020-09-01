from setuptools import setup, find_packages
import os

#CUDA = os.environ["CUDA"]
#TORCH = os.environ["TORCH"]

#if CUDA == None or TORCH == None:
#    print("Error: Environment variables $\{CUDA\} == {} and $\{TORCH\} == {} \n Please set these variables in your environment before continuing.".format(CUDA, TORCH))

#exit()

install_requires = [
#    "torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html".format({"CUDA":CUDA, "TORCH":TORCH}),
#    "torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html".format({"CUDA":CUDA, "TORCH":TORCH}),
#    "torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html".format({"CUDA":CUDA, "TORCH":TORCH}),
#    "torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html".format({"CUDA":CUDA, "TORCH":TORCH}),
#    "torch-geometric",
    #"networkx",
    "PosePrediction",
    "pandas"
]

dependency_links = [
    "git+https://github.com/AdamOlsson/PosePrediction"
]

packages = [
    "Datasets",
    "util/preprocessing",
]
# Install dependencies
setup(name='Exercise Prediction',
      version='0.1',
      description='Exercise prediction based on human graphs',
      author='Adam Olsson',
      #author_email='',
      #url='https://www.python.org/sigs/distutils-sig/',
      install_requires=install_requires,
      dependency_links=dependency_links,
      packages=find_packages(),
     )

# # Building PosePrediction
# print("\nBuilding paf lib...")
# stream = os.popen('python PosePrediction/setup.py install')
# output = stream.read()
# print(output)

