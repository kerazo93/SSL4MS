<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=False} -->

<!-- code_chunk_output -->

- [SSL4MS](#ssl4ms)
  - [Mass Spectrometry Data](#mass-spectrometry-data)
    - [Quick Overview](#quick-overview)
    - [Sources](#sources)
    - [Processing](#processing)
  - [Deep Learning for Tabular Data](#deep-learning-for-tabular-data)
  - [MS Data Corruption](#ms-data-corruption)

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
There's a more complete explanation of how the data was processed in the data directory of this repo, but here's the short version:
1. I parsed every MSP file and pulled out each spectrum that was provided regardless of what information it was missing. The only essential pieces that were needed were: (1) and InChI key for the compound, (2) a precursor ion mass, and (3) the MS2 spectra, pairs of masses and intensities. Each spectrum was saved as a CSV file: the first row is the precursor ion's mass with an intensity of zero, the susbsequent rows are the MS2 spectra sorted by decreasing mass.
2. The MS2 intensities were normalized so that the most intense peak had a normalized intensity of 100. All other fragment peaks' intensities are normalized to the max intensity. THe precursor ion always has an intesity of zero.
3. For every full length InChI key, I took the first block (the 2D connectivity portion) and created a JSON file for each unique compound. 
4. I used PubChem to find SMILES for each compound, then standardized them using RDKit. I also got compound formulas, bit and count fingerprints (Morgan, RDKit, Atom Pair, and Torsion) whenever possible.
5. For self supervised learning (**split 1**), I took all the spectra and their 2D InChI keys and used them to do a group split to generate train/val/test sets. I used an 80/10/10 split. In this case, it didn't matter if I was able to find SMILES, formulas, or fingerprints for the associated compounds.
6. For semi supervised learning (**split 2**), I removed spectra whose InChI keys I could not identify using PubChem. I took the remaining spectra and their associated 2D InChI keys to do a similar 80/10/10 train/val/test split.

## Deep Learning for Tabular Data
Others have pointed out that deep learning models have advanced tremendously for image (CNN) and sequence (Transformer) data. Following close behind are models for graph data (GNN) which are very common in cheminformatics. However, deep learning for tabular data are not as prevalent but I have found this [resource](https://github.com/wwweiwei/awesome-self-supervised-learning-for-tabular-data) that keeps track of developments in DL for tabular data.



## MS Data Corruption
The two columns (masses and intensities) need to be corrupted differently. For the masses, I have come up with a neutral loss based way of shifting the values:
* For each mass $m_i$ selected for corruption, replace it with the mass plus or minus a neutral loss $\delta_i$ that's smaller than $m_i$ to avoid generating negative masses. Put another way: $\tilde{m_i} = m_i \pm \delta_i, \quad \text{s.t.} \quad \delta_i<m_i$. This is the more straightforward approach.
* Here's a more complex approach: for each mass $m_i$ selected for corruption, find a larger unselected mass $m_j$ (so that $m_j > m_i$ and $j<i$ given the reverse sorting) and replace $m_i$ with $m_j$ plu sor minus a neutral loss $\delta_i$ that's smaller than $m_j$. Or: $\tilde{m_i} = m_j \pm \delta_i, \quad \text{s.t.} \quad \delta_i<m_j \quad \text{and} \quad m_j>m_i$.

For the intensities, the most straightforward way to corrupt them is to select $n$ random values to alter, find the sum of those values, and then generate another set of $n$ random values that will add up to the sum. In this way, the total intensity remains the same between the original and corrupted spectra.

I got the list of neutral losses from the [MS2Analyzer](https://pubs.acs.org/doi/10.1021/ac502818e) paper. The supporting information has an Excel file describing common neutral losses observed in metabolomics MS2 data.