# SEAD
Scene and Action Joint Prediction Model v1.0.

# Data preparation
Download [Action Genome annotations](https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys) and place them under ./annotations/.

Create the following three directories under the ./data/ directory:

* __corpus__:
This directory stores the corpus files for all objects. The corpus for each object is a file named "object.txt", such as "sofa.txt", in which, each line corresponds to a spatial-contact event sequence for a video (starting with "*" and ending with "#").

* __grammar__:
This directory stores the grammar files for all objects. Each object has an associated stochastic grammar file named "object.pcfg", such as "sofa.pcfg". These grammar files are learned from the following "Grammar Dictionary Learning" step.

* __duration__:
This directory stores the duration of spatial-contact events for all objects. Distinct spatial-contact events corresponding to each object are stored in a file named "object_duration.txt", such as "sofa_duration.txt", where each line corresponds to a spatial-contact event along with its average duration associated with the object.

# Grammar Dictionary Learning with ADIOS

<pre>
`ModifiedADIOS filename eta alpha context_size coverage`
</pre>

* filename: file name of each object corpus, such as "object.txt".
* eta: threshold of detecting divergence in the RDS graph, is set to 0.9 in our model.
* alpha: significance test threshold, usually is set to 0.01 in our model.
* context_size: size of the context window used for search for Equivalence Class, is set to 5 in our model.
* coverage: threhold for bootstrapping Equivalence classes, is set to 0.65 in our model.

The stochastic grammar learned by the AIDIOS algorithm will be stored in the "object.pcfg" file and placed in the "./data/grammar/".

# Scene Graph Prediction
python ./scene_predict.py

# Scene Graph Prediction
python ./action_anticipation.py






