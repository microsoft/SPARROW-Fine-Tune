`speciesnet_taxonomy.txt` is SpeciesNet's crop-model label file
(`data/model_package/always_crop_99710272_22x8_v12_epoch_00148.labels.txt` from
https://github.com/google/cameratrapai, Apache License 2.0). One line per output index:
`uuid;class;order;family;genus;species;common name`. Assisted labelling uses it to match
SpeciesNet predictions to a dataset's labels by common name, binomial, genus, family, order or class.
