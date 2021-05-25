Zexplorer SAGEBRAIN
==========================

This is the zEplorer Module for SAGEBRAIN.

Project Organization
------------

    ├── classes
    │   ├── author_class.py       <- Class that manages author information. The idea was to describe collections based on the authors but this was never used. 
    │   ├── document_class.py     <- Class that models the documents and information.
    │   ├── encoder_class.py      <- Class to manage the encoder model.
    │   └── session_class.py      <- Class to manage session information. A session is each ran of the system. The idea was to store the information from different sessions for different collections and users but this was never used.
    │   └── topic_class.py        <- Class to manage topics
    │   └── zotero_class.py       <- Class to manage zotero. Add your collection id in this class to conect to yours.
    │   └── nsaSrc            <- Original NSA repository
    │       ├── data    <- Makes src a Python module
    │           ├── external: Need to go to this directory and put the file downloaded from: http://magnitude.plasticity.ai/fasttext+approx/wiki-news-300d-1M.magnitude
    │           ├── extracted 
    │           ├── interim    
    │           ├── processed    
    │           ├── raw    
    │       ├── models


--------
