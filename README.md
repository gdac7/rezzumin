### Installation

* create a venv and activate it

* run
```bash
    pip install -r requirements.txt
```
```bash
    python -m spacy download pt_core_news_sm
```
* download word2vec 'CBOW 100 dimensões' in http://www.nilc.icmc.usp.br/embeddings
* put model ```cbow_s100.txt``` in ```db/``` folder
* change directory to ```code/```
```bash
    cd code
```
* run
```bash
    python -m spacy init vectors pt ..\db\cbow_s100.txt vectors_spacy
```
* everything is set up !
