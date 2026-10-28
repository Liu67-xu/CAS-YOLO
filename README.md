CAS-YOLO: annotations, trained weights and code
=============================================

Companion material for

  "CAS-YOLO: a detector with residual-branch coordinate attention, adaptive
   spatial feature fusion and background auxiliary learning for small-object
   detection of cucumber growth stages"

The study detects four cucumber growth stages (developing, maturing, spotting,
rotting) on 4,274 images with 20,067 annotated instances.

Everything needed to repeat the experiments is in this repository, apart from
the source images, which are third-party material and are therefore not
redistributed (section 3).


0. Getting the material
-----------------------

    git clone https://github.com/Liu67-xu/CAS-YOLO.git
    cd CAS-YOLO

The repository is the working copy, so every file can be browsed, opened and
diffed on its own.

Python 3.11 and about 20 MB of disk are needed for the code and the weights.
The images are added separately, see section 3.


Quick start
-----------

    pip install ultralytics==8.3.163   # then copy code/ultralytics/ over the installed package
    python code/set_path.py            # point the dataset definition at this repository
    yolo val model=weights/CAS-YOLO_best.pt data=code/configs/cucumber_dataset.yaml imgsz=640 batch=20

The first line is not optional: C2f_Att, ASFF and CA are not part of stock
ultralytics, so without the copied files the weights cannot be unpickled.
Section 4 gives the same three steps with the commands for Windows and Linux,
the expected output of the two validations, and the tolerance to accept.
Section 3 explains how to obtain the images, which are not redistributed here.


1. Contents
-----------

data/labels/train1/     3,332 files   Training split used in the paper (15,851 instances)
data/labels/train/      5,332 files   BG-Aux training set: the above plus 2,000
                                      target-free background patches, most of them
                                      paired with empty label files, which act as
                                      negative samples during training
data/labels/valid/        567 files   Validation split (2,353 instances)
data/labels/test/         375 files   Held-out test split (1,863 instances), not
                                      used during model development

data/splits/train1.txt, train.txt, valid.txt, test.txt
                                      Exact file lists of each split, one image
                                      name per line. These fix the 567-image
                                      validation split, so the reported numbers
                                      can be reproduced on exactly the same data.

data/image_sources.csv                One row per image (4,274): file name, split
                                      and source group.
data/SOURCE_GROUPS.csv                The four source groups with project page and
                                      licence of each. See section 3.

weights/CAS-YOLO_best.pt              Weights of the proposed detector
                                      (mAP@50 = 90.39 %, mAP@50-95 = 70.18 %)
weights/YOLOv8n_baseline_best.pt      YOLOv8n baseline trained on the same train1
                                      split under an identical protocol
                                      (mAP@50 = 89.82 %, mAP@50-95 = 66.05 %);
                                      this is the reference row of the comparison
                                      table

code/configs/yolov8_C2fAtt_CA_asff.yaml   Model definition: C2f_Att with
                                      coordinate attention in the backbone,
                                      ASFF in the neck
code/configs/cucumber_dataset.yaml    Dataset definition (edit `path:` once)
code/set_path.py                      Writes the repository location into it
code/ultralytics/nn/modules/block.py  C2f_Att: coordinate attention embedded in
                                      the C2f residual branches ("CA" argument)
code/ultralytics/nn/Attmodules/       ASFF (adaptive spatial feature fusion) and
                                      the coordinate-attention block
code/scripts/extract_background.py    BG-Aux: background patch extraction from
                                      target-free regions of the training images
code/scripts/merge_background.py      BG-Aux: merging those patches into train1,
                                      producing the train split
code/train_example.py                 Training script reproducing the setup
environment.txt                       Versions actually used
VERSION.txt                           What changed with respect to the first upload
checksums.txt                         SHA-256 of every file in this repository
requirements.txt                      Pinned versions of the packages used
CITATION.cff                          Machine-readable citation metadata
THIRD_PARTY_NOTICES.md                Licence of the files derived from Ultralytics
LICENSE                               AGPL-3.0, see section 7
LICENSE-annotations                   CC BY 4.0 for annotations and weights


2. Annotation format and class distribution
-------------------------------------------

Standard YOLO format, one .txt per image, one line per instance,

    <class_id> <x_center> <y_center> <width> <height>

with all coordinates normalised to [0, 1]:

    0 = developing_cucumber   1 = maturing_cucumber
    2 = spotting_cucumber     3 = rotting_cucumber

A label file shares the base name of its image (abc123.txt labels abc123.jpg).
An empty .txt marks an image without any target.

    split     images  labelled  empty  instances  developing  maturing  spotting  rotting
    train1     3,332     3,321     11     15,851      11,316     2,816       681     1,038
    valid        567       566      1      2,353       1,358       782       111       102
    test         375       375      0      1,863       1,348       452        42        21

The strong imbalance towards the developing stage is a property of the crop and
is analysed in the paper.


3. The source images are not included
-------------------------------------

The images come from four groups, listed in data/SOURCE_GROUPS.csv together with
the project page and the licence of each. Three of them are public Roboflow
Universe datasets released under CC BY 4.0 by their respective owners. The
fourth consists of greenhouse cucumber images collected from the web and
re-annotated by the authors; as it was assembled from individual web pages
rather than from one dataset, it has no single project page, and its files keep
the numbering of the annotation export (D1_S_0001 - D1_S_2235). In all four
cases the photographs are owned by others, so they are not redistributed here;
what is provided is the four-stage annotation set, which the authors produced or
corrected.

data/image_sources.csv lists every expected file name with its split and its
source group, so the image set can be reassembled from the original providers
and matched to the released annotations by file name. data/splits/*.txt then fix
the exact membership of each split. A portion of the instances was annotated by
the authors, and the annotations supplied with the public datasets were reviewed
and corrected so that every label follows the single four-stage protocol defined
in the paper.


4. Reproducing the reported numbers
-----------------------------------

Step 1 - patch the installed package, otherwise the weights cannot be unpickled
(the custom layers C2f_Att and ASFF are not part of stock ultralytics):

    pip install ultralytics==8.3.163
    cp -r code/ultralytics/* <site-packages>/ultralytics/

Step 2 - put the images under data/images/<split>/ and point the dataset
definition at this repository:

    python code/set_path.py

Step 3 - validate:

    yolo val model=weights/CAS-YOLO_best.pt \
             data=code/configs/cucumber_dataset.yaml \
             imgsz=640 batch=20

    ->  mAP@50 ~90.39 %   mAP@50-95 ~70.18 %

    yolo val model=weights/YOLOv8n_baseline_best.pt \
             data=code/configs/cucumber_dataset.yaml \
             imgsz=640 batch=20

    ->  mAP@50 ~89.82 %   mAP@50-95 ~66.05 %

Please keep batch=20. The validator applies a rectangular letterbox per batch, so
a different batch size shifts the reported values by a few hundredths of a
point: batch=16, for example, gives 90.35 / 70.11 for the same weights. Values
reproduce to within about 0.1 pp; residual differences of that order may also
appear across GPU models, CUDA builds and driver versions.


5. Training configuration used for the reported results
-------------------------------------------------------

    model      code/configs/yolov8_C2fAtt_CA_asff.yaml
    data       code/configs/cucumber_dataset.yaml
    epochs     100 (patience 30)      batch 20        imgsz 640
    optimizer  auto, lr0 0.01, lrf 0.01, momentum 0.937, weight decay 0.0005
    aug        degrees 8.0, translate 0.08, scale 0.5, fliplr 0.5, mosaic 0.7,
               close_mosaic 15, mixup 0.0, copy_paste 0.0
    misc       pretrained true, seed 0, AMP on, cache ram

    python code/train_example.py


6. Notes
--------

All comparisons reported in the paper were carried out under one common
training-validation protocol; the test split was kept untouched throughout. The
BG-Aux training split can be rebuilt from train1 with
code/scripts/extract_background.py followed by code/scripts/merge_background.py.

7. Licence and citation
-----------------------

The repository as a whole is released under the GNU Affero General Public
License v3.0, because it distributes modified files of Ultralytics YOLOv8,
which is itself AGPL-3.0; see LICENSE and THIRD_PARTY_NOTICES.md. The files
written by the authors are additionally available under the MIT licence, at the
recipient's option. The annotations in data/labels/ and the two sets of weights
in weights/ are released under CC BY 4.0; see LICENSE-annotations. The source
images are third-party material and are not redistributed here.

If you use this material, please cite the accompanying manuscript;
CITATION.cff holds the same information in machine-readable form.
