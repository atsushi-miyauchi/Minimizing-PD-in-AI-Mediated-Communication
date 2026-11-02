# Data license, provenance, and citations

This notice applies to the graph files under `data/raw/`. It does not apply to
the software, which is released under the MIT License in `LICENSE`.

## License

The graph files were obtained from Network Repository. Its data policy states
that all data are licensed under a Creative Commons Attribution-ShareAlike
(CC BY-SA) License and permits sharing and adaptation subject to attribution,
ShareAlike, preservation of notices, and dataset-specific citation requests:

https://networkrepository.com/policy.php

The Network Repository policy does not identify a license version. We
therefore do not assign one here; recipients should consult the policy above.
Each graph remains subject to that policy and any rights or notices associated
with its original source. Redistribution from this repository does not imply
endorsement by Network Repository or the original data creators.

## Network Repository attribution

Please cite:

> Ryan A. Rossi and Nesreen K. Ahmed. “The Network Data Repository with
> Interactive Graph Analytics and Visualization.” AAAI, 2015.
> https://networkrepository.com/

## Dataset-specific sources

| Bundled file | Network Repository page | Requested source citation |
|---|---|---|
| `soc-wiki-Vote.mtx` | https://networkrepository.com/soc_wiki_Vote.php | Leskovec, Huttenlocher, and Kleinberg (2010) |
| `soc-hamster.edges` | https://networkrepository.com/soc-hamsterster.php | Hamsterster social network |
| `soc-advogato.edges` | https://networkrepository.com/soc_advogato.php | Massa, Salvetti, and Tomasoni (2009) |
| `soc-epinions.mtx` | https://networkrepository.com/soc-epinions.php | Richardson, Agrawal, and Domingos (2003); Massa and Avesani (2005) |
| `soc-Slashdot0902.mtx` | https://networkrepository.com/soc-Slashdot0902.php | Network Repository; the dataset page identifies J. Leskovec, K. Lang, A. Dasgupta, and M. Mahoney as authors |
| `soc-youtube.mtx` | https://networkrepository.com/soc-youtube.php | Mislove et al. (2007) |
| `additional/soc-karate.mtx` | https://networkrepository.com/soc-karate.php | Zachary (1977) |
| `additional/soc-dolphins.mtx` | https://networkrepository.com/soc-dolphins.php | Lusseau et al. (2003) |

Full references follow.

- Jure Leskovec, Daniel Huttenlocher, and Jon Kleinberg. “Signed Networks in
  Social Media.” CHI, 2010.
- Hamsterster. “Hamsterster Social Network.” http://www.hamsterster.com/
- Paolo Massa, Martino Salvetti, and Danilo Tomasoni. “Bowling Alone and Trust
  Decline in Social Network Sites.” DASC, 2009.
- Matthew Richardson, Rakesh Agrawal, and Pedro Domingos. “Trust Management
  for the Semantic Web.” ISWC, 2003.
- Paolo Massa and Paolo Avesani. “Controversial Users Demand Local Trust
  Metrics: An Experimental Study on epinions.com Community.” AAAI, 2005.
- Alan Mislove, Massimiliano Marcon, Krishna P. Gummadi, Peter Druschel, and
  Bobby Bhattacharjee. “Measurement and Analysis of Online Social Networks.”
  IMC, 2007.
- Wayne W. Zachary. “An Information Flow Model for Conflict and Fission in
  Small Groups.” Journal of Anthropological Research, 1977.
- David Lusseau, Karsten Schneider, Oliver J. Boisseau, Patti Haase, Elisabeth
  Slooten, and Steve M. Dawson. “The Bottlenose Dolphin Community of Doubtful
  Sound Features a Large Proportion of Long-Lasting Associations.” Behavioral
  Ecology and Sociobiology, 2003.

## Extraction, naming, and runtime preprocessing

- Files were extracted from Network Repository download archives and placed
  under `data/raw/`. The Hamsterster filename was shortened to
  `soc-hamster.edges`; the dataset itself is `soc-hamsterster` on Network
  Repository.
- No intentional edge additions or deletions were made to the stored files.
  Existing Matrix Market and SuiteSparse notices were retained.
- At runtime, the loader keeps the largest weakly connected component, expands
  undirected edges in both directions, and constructs a row-stochastic matrix.
  Directedness and weight handling are recorded in each YAML config. These are
  analysis-time transformations; the bundled raw files are not overwritten.

Network Repository notes that other rights, including privacy, publicity, and
moral rights, may still apply and provides the data without warranty. See its
policy and disclaimer for details.
