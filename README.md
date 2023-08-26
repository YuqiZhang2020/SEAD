# SEAD
==
Scene and Action Joint Prediction Model v1.0.

# Data preparation
Download [Action Genome annotations](https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys) and place them under ./annotations.

Create the following three directories under the ./data directory:

* corpus:
The corpus for each object is a file named "object.txt", such as "sofa.txt", in which, each line corresponds to a spatial-contact event sequence (beginning with "*", and concluding with "#") for a video.

* grammar:
The stochastic grammar associated with each object is a file named "object.pcfg", such as "sofa.pcfg" (These grammar files learn from the following "Grammar Diction Learning" step.

* duration:

Distinct spatial-contact events corresponding to each object are stored in a file named "object_duration.txt", such as "sofa_duration.txt", in which, each line corresponds to a spatial-contact event along with its average duration associated with the object.

# Grammar Diction Learning with ADIOS (./madios)
 --> object.pcfg
Usage:
ModifiedADIOS <filename> <eta> <alpha> <context_size> <coverage> ---OPTIONAL--- <number_of_new_sequences>

<filename>,     file name of each object corpus, such as "object.txt"
<eta>,          threshold of detecting divergence in the RDS graph, is set to 0.9 in our model.
<alpha>,        significance test threshold, usually is set to 0.01 in our model.
<context_size>, size of the context window used for search for Equivalence Class, is set to 5 in our model.
<coverage>,     threhold for bootstrapping Equivalence classes, is set to 0.65 in our model.






