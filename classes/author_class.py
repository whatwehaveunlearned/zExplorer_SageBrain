# -*- coding: utf-8 -*-
import pandas as pd
import pdb
from pybliometrics.scopus import AuthorSearch, config
# config["Authentication"]["APIKey"] = "5453f9daab7dc6734b60da63475a7b58"
# scopus = Scopus(key)

#Author Class
class Author:
    """ Author Class """
    def __init__(self, firstName,lastName):
        self.firstName = firstName
        self.lastName = lastName
        self.documents = []
        
    def search_data(self):
        pdb.set_trace()
        s = AuthorSearch('AUTHLAST(Selten) and AUTHFIRST(Reinhard)', refresh=True)
        pdb.set_trace()