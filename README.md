# Perceptron-and-Decision-Tree

Il progetto permette di confrontare le curve di apprendimento dei modelli Perceptron e Decision Tree, utilizzati per classificare le immagini di un dataset di articoli di moda, disponibile a questo [link](https://github.com/zalandoresearch/fashion-mnist). 

# Prima dell'esecuzione 

Le librerie necessarie per la corretta esecuzione del programma sono le seguenti: 

- [os](https://docs.python.org/3.7/library/os.html), [shutil](https://docs.python.org/3.7/library/shutil.html), [tempfile](https://docs.python.org/3.7/library/tempfile.html) : Sono i moduli che permettono, tra le altre cose, di accedere alla cartella del progetto, creare al suo interno nuove cartelle, costruire cartelle temporanee e spostare i file da una cartella all'altra. Queste operazioni sono fondamentali per ottenere il dataset senza doverlo scaricare manualmente dal [link](https://github.com/zalandoresearch/fashion-mnist) o dover clonare l'intero progetto github di Zalando. Infatti la procedura per disporre del dataset e soprattutto del file di parsing è incapsulata nella funzione ``loadDataset``, presente nel file **functions.py**, che al momento della prima esecuzione provvede a scaricare quanto detto e a salvarlo nella cartella **./dataset**. 

- [GitPython](https://gitpython.readthedocs.io/en/stable/) : E' il modulo che permette di clonare da un repository  github le cartelle e/o i file di cui si ha bisogno. Nel nostro caso permette di clonare la cartella che contiene i dati di training e di testing dal repository di Zalando, insieme al file di parsing del dataset stesso. 

- [Scikit-Learn](https://scikit-learn.org/stable/) : E' la libreria che contiene i modelli e le funzioni necessarie all'allenamento e alla predizione dei dati, oltre alle metriche per valutarne le prestazioni. 

- [Numpy](https://numpy.org/) : E' la libreria che contiene funzioni generiche per lavorare con array, sequenze di indici e molto altro.

- [Matplotlib](https://matplotlib.org/) : E' la libreria che fornisce le funzioni per disegnare i grafici e tutte le relative impostazioni. Mette a disposizione anche delle funzioni per stampare a video le immagini, che nel nostro caso sono state utilizzate nella costruzione delle funzioni per visualizzare le immagini del dateset. 

Tutti i moduli elencati possono essere scaricati attraverso il package installer [pip](https://pip.pypa.io/en/stable/), semplicemente digitando da riga di comando ``pip install <nome_modulo>``.

# Codice 

Una volta importate tutte le librerie necessarie, si può passare ad eseguire il codice di testing, concentrato nel file **main.py**. Il programma carica il dataset, normalizza i dati, istanzia i modelli e disegna le rispettive curve di apprendimento. E' ovviamente possibile modificare alcuni parametri dell'esecuzione, come per esempio gli iperparametri dei modelli o il numero di volte (che corrisponde al contenuto della variabile ``iter``) in cui si richiede di ripetere il calcolo dei valori di accuratezza sul training e sul testing, al variare della dimensione del training set, per poi calcolarne la media e la deviazione standard e mostrarli nel grafico della curva di apprendimento. Tutte le altre funzioni ausiliarie sono contenute nel file **functions.py**. 

Per avere maggiori informazioni sulla logica con cui sono stati condotti gli esperimenti e sull'interpretazione dei risultati ottenuti si rimanda alla lettura del file **Relazione progetto AI.pdf**, presente nel repository. 
