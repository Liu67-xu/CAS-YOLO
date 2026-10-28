Third-party notices
===================

Ultralytics YOLOv8 (AGPL-3.0)
-----------------------------

The following files are modified copies of files from the Ultralytics YOLOv8
package (https://github.com/ultralytics/ultralytics), which is distributed
under the GNU Affero General Public License, version 3.0:

    code/ultralytics/nn/modules/block.py
    code/ultralytics/nn/modules/__init__.py
    code/ultralytics/nn/tasks.py

Modifications made by the authors of this repository: addition of the C2f_Att
block with coordinate attention; registration of the ASFF and
coordinate-attention modules under nn/Attmodules; and the parser changes needed
to build the CAS-YOLO architecture from code/configs/yolov8_C2fAtt_CA_asff.yaml.
The original licence headers of these files are kept intact.

These files remain under AGPL-3.0 and are not relicensed by this repository.

Effect on the licence of this repository
----------------------------------------

Because the repository distributes the files listed above, the repository as a
whole is released under AGPL-3.0; see LICENSE. The files written by the authors

    code/configs/
    code/scripts/
    code/set_path.py
    code/train_example.py
    code/ultralytics/nn/Attmodules/

are additionally available under the MIT licence, at the recipient's option.

Annotations and model weights
-----------------------------

The four-stage annotations in data/labels/ and the trained weights in weights/
are released under CC BY 4.0; see LICENSE-annotations.

Source images
-------------

Not distributed with this repository. The images come from four third-party
groups, each of which keeps its own licence; see data/SOURCE_GROUPS.csv and
data/image_sources.csv.
