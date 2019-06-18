# LTG-Oslo Hierarchical Multi-task Network: The importance of negation for document-level sentiment in Spanish

Jeremy Barnes [jeremycb@ifi.uio.no]

This repository contains the multi-task hierarchical model from

Jeremy Barnes. 2019. **LTG-Oslo Hierarchical Multi-task Network: The importance of negation for document-level sentiment in Spanish**. In *Proceedings of NEGES 2019: Workshop on Negation in Spanish*.

## Models
1. Bag-of-Words + L2 regularized Logistic Regression
2. Hierarchical Single-task Model
3. Hierarchical Multi-task Model

## Dataset
[SFU ReviewSP-Neg](https://link-springer-com.ezproxy.uio.no/article/10.1007/s10579-017-9391-x)

If you use this dataset please cite the following paper:

Jiménez-Zafra, S. M., Taulé, M., Martín-Valdivia, M. T., Ureña-López, L. A., & Martí, M. A. (2018c). **SFU Review SP-NEG: a Spanish corpus annotated with negation for sentiment analysis. A typology of negation patterns**. *Language Resources and Evaluation*, 52(2), 533-569. First published online May 22, 2017.


### Requirements

1. Python 3
2. sklearn  ```pip install -U scikit-learn```
3. pytorch ```pip install torch```

### Data you need
1. Word embeddings ([available here](https://drive.google.com/open?id=1mUFZvGevO4vaktUFwpgeoyz-joxuljom))
	- Download and unzip them in directory /embeddings


### Running the program

To reproduce the results from the paper,

```
git clone https://github.com/jbarnesspain/neges_2019
cd neges_2019/
mkdir embeddings
```

Then download the [embeddings](https://drive.google.com/open?id=1mUFZvGevO4vaktUFwpgeoyz-joxuljom) and unzip them in the /embeddings directory. Then run the hierarchical model:

```
cd models
python3 hierarchical_training.py
```

### Output

The best models will be saved in saved_models

When you run test_model.py, it will give you the mean and standard deviation over 5 runs.

```
python3 test_model.py
```

### Reference

```
@inproceedings{Barnes2019,
  author = {Barnes, Jeremy},
  title = {LTG-Oslo Hierarchical Multi-task Network: The importance of negation for document-level sentiment in Spanish},
  booktitle = {Proceedings of NEGES 2019: Workshop on Negation in Spanish},
  year = {2019},
  address = {Bilbao, Spain}
}
```
