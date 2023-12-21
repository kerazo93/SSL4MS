<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=False} -->

<!-- code_chunk_output -->

- [SSL4MS](#ssl4ms)
  - [Mass Spectrometry Data](#mass-spectrometry-data)
    - [Quick Overview](#quick-overview)
    - [Sources](#sources)
    - [Processing](#processing)
  - [Deep Learning and Tabular Data](#deep-learning-and-tabular-data)

<!-- /code_chunk_output -->


# SSL4MS
[![DOI](https://zenodo.org/badge/721344082.svg)](https://zenodo.org/doi/10.5281/zenodo.10419946)

SSL4MS is short for self/semi-supervised learning for (small molecule) mass spectrometry data. Here I view processed MS data as tabular data and explore various forms of self and semi supervised learning for this particular type of experimental data.


## Mass Spectrometry Data

### Quick Overview
Whenever you see "mass spectrometry data" here (or in a lot of other places), what most people are referring to is data generated from samples analyzed by [LC-MS/MS](https://en.wikipedia.org/wiki/Liquid_chromatography–mass_spectrometry): liquid chromatography followed by tandem mass spectrometry to yield MS2 (or fragment) spectra. Very broadly and without getting too into the technical weeds, this is how this type of data would be generated:
1. An aliquot of a sample containing some analytes (could be small molecules, proteins, DNA, literally anything) is injected into an LC system and passed through a chromatography column to separate the analytes based on some physical property (usually polarity).
2. As the analytes elute form the column, they're passed directly into an ionization source which will (you guessed it) bombard neutral molecules with electrons to geneate ions; in the process some neutral molecules may form adducts with surrounding substances and then the whole thing gets ionized.
3. Ions are then accelerated through the mass spectrometer and their masses are measured with great precision. Sometimes the intital ions (also called **precursor ions**) are fragmented and then their associated fragments' masses are also measured.
4. The resulting data can be anayzed to generate a compound's mass spectrum which includes (minimally) the precursion ion's mass along with the fragment ions' masses and their corresponding intensities.

Although there are lots of ways to represent mass spectometry data out there, I think it lends itself very easily to a tabular format with two columns, one for masses and one for intensities, where each row is either the precursor ion or its fragments. This is the data representation I explore here.

### Sources
Small disclaimer here: I am only looking at mass spectra from small molecules which encompasses many types of compounds. Critically, I exclude peptides because there are plenty of sophisticated tools out there for proteomics research but metabolomics research is still lagging here. Also, I have not removed lipids or carbohydrates from the datasets I have compiled but these are two notoriously challenging groups of small molecules for mass spectrometry due to being made up of isomeric monomers.

That being said, I scrapped the internet and pulled together the following data sources:
* Mass Bank of North America ([MoNa](https://mona.fiehnlab.ucdavis.edu/downloads)): I took the LC-MS/MS Spectra MSP file.
* Global Natural Products Social Molecular Networking ([GNPS](https://gnps-external.ucsd.edu/gnpslibrary)): I took the ALL_GNPS_NO_PROPOGATED MSP file.
* Mass Bank of Europe ([MoE](https://github.com/MassBank/MassBank-data/releases/)): I took the latest (2023.11) release of the NIST and RIKEN MSP files.

### Processing
There's a more complete explanation of how the data was processed in the data directory of htis repo, but here's the short version:
1. 
2. 

## Deep Learning and Tabular Data