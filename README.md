# Neural Body Fitting code repository

![example_output](/demo/up/examples.png)

## Setup:
* `git clone --recursive http://github.com/mohomran/neural_body_fitting`
* create and activate a fresh virtualenv
* `pip install tensorflow-gpu==1.6.0` (or `tensorflow==1.6.0`)
* inside the root folder run `pip install -r requirements.txt`
* navigate to `external/up` and run `python setup.py develop` (which will install the UP toolbox)
* download SMPL (at http://smpl.is.tue.mpg.de/downloads) and unzip to `external/`
* download the [segmentation model](http://transfer.d2.mpi-inf.mpg.de/mohomran/nbf/refinenet_up.tgz) and extract into `models/`
* download the [fitting model](http://transfer.d2.mpi-inf.mpg.de/mohomran/nbf/demo_up.tgz) and extract into `experiments/states`

## Demo:
The following command will perform inference on 60 images from the UP dataset:

```
python run.py infer_segment_fit experiments/config/demo_up/ \
              --inp_fp demo/up/input/\
              --out_fp demo/up/output\
              --visualise render
```

The results can be viewed by opening the file `demo/up/output/index.html` in a browser. These were selected to demonstrate both success and failure cases. Most of the processing time (~80%) is taken up by the mesh renderer. Alternatively, you can use ```--visualise pose``` which is quicker and just plots the projected SMPL joints.

## Training:
Coming Soon

## Citation
If you find any parts of this code useful, please cite the following [paper](https://arxiv.org/abs/1808.05942):
```
@inproceedings {omran2018nbf,
  title = {Neural Body Fitting: Unifying Deep Learning and Model-Based Human Pose and Shape Estimation},
  journal = {International Conference on 3D Vision (3DV)},
  year = {2018},
  author = {Omran, Mohamed and Lassner, Christoph and Pons-Moll, Gerard and Gehler, Peter V. and Schiele, Bernt}
  address = {Verona, Italy},
}
```

## Acknowledgements
The repository is modelled after (and partially adopts code from) Christoph Lassner's [Generating People](https://github.com/classner/generating_people) project.
The example data provided is from his [Unite the People](http://files.is.tuebingen.mpg.de/classner/up/) dataset.
